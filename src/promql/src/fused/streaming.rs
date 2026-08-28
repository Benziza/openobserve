// Copyright 2026 OpenObserve Inc.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

//! Streaming evaluation of `agg(range_func(selector))` over hash-sorted
//! metrics files: each hash band merges its hash-ordered file chains one
//! series at a time; a series is folded into its group accumulator and
//! dropped, so the sample matrix is never materialized.

use std::{hash::Hasher, sync::Arc, time::Duration};

use config::{
    TIMESTAMP_COL_NAME,
    meta::promql::{
        EXEMPLARS_LABEL, HASH_LABEL, NAME_LABEL, STREAMING_AGG_TABLE_SUFFIX, VALUE_LABEL,
        value::{EvalContext, ExtrapolationKind, Label, Labels, RangeValue, Sample, Value},
    },
    utils::hash::gxhash,
};
use datafusion::{
    arrow::{
        array::{AsArray, RecordBatch},
        datatypes::{DataType, Float64Type, Int64Type, Schema, UInt64Type},
    },
    error::{DataFusionError, Result},
    execution::{SendableRecordBatchStream, TaskContext},
    physical_plan::{
        ExecutionPlan, execute_stream, execute_stream_partitioned,
        expressions::Column,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
    prelude::{DataFrame, SessionContext, col, lit},
};
use futures::TryStreamExt;
use hashbrown::{HashMap, hash_map::Entry};
use infra::errors::{Error, ErrorCodes};
use promql_parser::{label::Matchers, parser::LabelModifier};

use super::{accumulator::FusedAccumulator, eval::fold_series, op::FusedAggOp};
use crate::{
    functions::{KEEP_METRIC_NAME_FUNC, RangeFunc},
    load_series::{LabelColumn, apply_time_window, batch_run_len},
    utils::apply_matchers,
};

type GroupAccs = HashMap<u64, GroupEntry>;

/// Load-window and matcher parameters of one selector scan, with the window
/// already shifted for any `offset` modifier.
pub(crate) struct StreamingSelector<'a> {
    pub table_name: &'a str,
    pub matchers: &'a Matchers,
    pub start: i64,
    pub end: i64,
    pub step: i64,
    pub lookback: i64,
    pub offset: i64,
}

/// The `agg(range_func(...))` pair being evaluated.
pub(crate) struct FusedShape {
    pub op: FusedAggOp,
    pub func: Arc<dyn RangeFunc>,
    pub range: Duration,
}

struct FoldParams {
    op: FusedAggOp,
    func: Arc<dyn RangeFunc>,
    counter_kind: Option<ExtrapolationKind>,
    range: Duration,
    offset: i64,
    eval_ctx: EvalContext,
    timestamps: Vec<i64>,
    group_cols: Vec<String>,
}

struct GroupEntry {
    labels: Labels,
    acc: FusedAccumulator,
}

/// One hash-ordered input of a band's merge (a chain of non-overlapping
/// files). All samples of one series sit in a single run per chain.
struct ChainCursor {
    stream: SendableRecordBatchStream,
    batch: Option<RecordBatch>,
    row: usize,
}

impl ChainCursor {
    async fn start(stream: SendableRecordBatchStream) -> Result<Self> {
        let mut cursor = Self {
            stream,
            batch: None,
            row: 0,
        };
        cursor.next_batch().await?;
        Ok(cursor)
    }

    fn head_hash(&self) -> Option<u64> {
        let batch = self.batch.as_ref()?;
        Some(batch[HASH_LABEL].as_primitive::<UInt64Type>().values()[self.row])
    }

    async fn next_batch(&mut self) -> Result<()> {
        self.row = 0;
        loop {
            match self.stream.try_next().await? {
                Some(batch) if batch.num_rows() == 0 => continue,
                batch => {
                    self.batch = batch;
                    return Ok(());
                }
            }
        }
    }

    /// Appends this chain's samples of series `hash`, following the run across
    /// batch boundaries until the hash changes or the chain ends.
    async fn consume_run(
        &mut self,
        hash: u64,
        offset: i64,
        samples: &mut Vec<Sample>,
    ) -> Result<()> {
        while let Some(batch) = &self.batch {
            let hashes = batch[HASH_LABEL].as_primitive::<UInt64Type>().values();
            if hashes[self.row] != hash {
                return Ok(());
            }
            let run_len = batch_run_len(hashes, self.row);
            let times = batch[TIMESTAMP_COL_NAME]
                .as_primitive::<Int64Type>()
                .values();
            let values = batch[VALUE_LABEL].as_primitive::<Float64Type>().values();
            samples.extend(
                times[self.row..self.row + run_len]
                    .iter()
                    .zip(&values[self.row..self.row + run_len])
                    .map(|(&timestamp, &value)| Sample::new(timestamp + offset, value)),
            );
            self.row += run_len;
            if self.row < hashes.len() {
                return Ok(());
            }
            self.next_batch().await?;
        }
        Ok(())
    }
}

/// Runs the fused aggregation as parallel per-hash-band ordered streams.
/// Returns `None` when the layout or query shape rules the streaming plan out
/// (no ordered table, non-UInt64 hashes, `without` grouping, a plan that would
/// need an actual sort); the caller then falls back to the materializing path.
pub(crate) async fn streaming_fused_agg(
    ctx: &SessionContext,
    schema: &Schema,
    selector: StreamingSelector<'_>,
    shape: FusedShape,
    modifier: &Option<LabelModifier>,
    eval_ctx: &EvalContext,
    timeout: u64,
) -> Result<Option<Value>> {
    let start_time = std::time::Instant::now();
    let trace_id = eval_ctx.trace_id.clone();

    if schema
        .field_with_name(HASH_LABEL)
        .is_ok_and(|field| field.data_type() != &DataType::UInt64)
    {
        return Ok(None);
    }
    let Some(group_cols) = group_label_columns(modifier, schema, shape.func.name()) else {
        return Ok(None);
    };
    let sorted_table = format!("{}{STREAMING_AGG_TABLE_SUFFIX}", selector.table_name);
    let Ok(df) = ctx.table(sorted_table.as_str()).await else {
        return Ok(None);
    };

    let df = apply_time_window(
        df,
        selector.start,
        selector.end,
        selector.step,
        selector.lookback,
    )?;
    let df = apply_matchers(df, selector.matchers)?;

    let mut columns = vec![TIMESTAMP_COL_NAME, HASH_LABEL, VALUE_LABEL];
    columns.extend(group_cols.iter().map(String::as_str));

    let bands = ctx.state().config().target_partitions();
    let Some((band_inputs, band0_plan)) =
        build_band_inputs(&df, &columns, bands, &trace_id).await?
    else {
        return Ok(None);
    };

    log::info!(
        "[trace_id: {trace_id}] [PromQL Timing] streaming fused {}({}) started with {bands} bands",
        shape.op.name(),
        shape.func.name(),
    );
    let params = Arc::new(FoldParams {
        op: shape.op,
        func: shape.func.clone(),
        counter_kind: shape.func.counter_extrapolation(),
        range: shape.range,
        offset: selector.offset,
        eval_ctx: eval_ctx.clone(),
        timestamps: eval_ctx.timestamps(),
        group_cols,
    });
    let folds = run_bands(band_inputs, params.clone(), timeout).await?;

    if config::get_config().common.print_key_sql {
        log::info!(
            "[trace_id: {trace_id}] [PromQL] streaming band 0 metrics:\n{}",
            datafusion::physical_plan::display::DisplayableExecutionPlan::with_metrics(
                band0_plan.as_ref()
            )
            .indent(true)
        );
    }
    let series_count: usize = folds.iter().map(|(_, series)| series).sum();
    let value = merge_folds(
        folds.into_iter().map(|(groups, _)| groups).collect(),
        &params.timestamps,
    );
    log::info!(
        "[trace_id: {trace_id}] [PromQL Timing] streaming fused {}({}) completed in {:?}, folded {series_count} series into {} series",
        shape.op.name(),
        shape.func.name(),
        start_time.elapsed(),
        match &value {
            Value::Matrix(matrix) => matrix.len(),
            _ => 0,
        },
    );
    Ok(Some(value))
}

/// Columns the aggregation groups by, sorted for a stable label order.
/// `None` when the grouping cannot be resolved to a column set (`without`).
fn group_label_columns(
    modifier: &Option<LabelModifier>,
    schema: &Schema,
    func_name: &str,
) -> Option<Vec<String>> {
    let include = match modifier {
        None => return Some(vec![]),
        Some(LabelModifier::Include(labels)) => &labels.labels,
        Some(LabelModifier::Exclude(_)) => return None,
    };
    let mut cols: Vec<String> = include
        .iter()
        .filter(|name| {
            let name = name.as_str();
            name != TIMESTAMP_COL_NAME
                && name != HASH_LABEL
                && name != VALUE_LABEL
                && name != EXEMPLARS_LABEL
                // range functions strip the metric name before aggregation
                && (name != NAME_LABEL || KEEP_METRIC_NAME_FUNC.contains(func_name))
                && schema.field_with_name(name).is_ok()
        })
        .cloned()
        .collect();
    cols.sort();
    cols.dedup();
    Some(cols)
}

/// Uniform partition of the u64 hash space into `count` inclusive ranges.
fn hash_bands(count: usize) -> Vec<(u64, u64)> {
    let count = count.max(1) as u128;
    let span = (u64::MAX as u128) + 1;
    (0..count)
        .map(|band| {
            let lo = (span * band / count) as u64;
            let hi = (span * (band + 1) / count - 1) as u64;
            (lo, hi)
        })
        .collect()
}

/// Builds every band's ordered input streams; `None` (with the offending band
/// logged) means some band's plan cannot stream in order.
async fn build_band_inputs(
    df: &DataFrame,
    columns: &[&str],
    bands: usize,
    trace_id: &str,
) -> Result<Option<(Vec<Vec<SendableRecordBatchStream>>, Arc<dyn ExecutionPlan>)>> {
    let mut band_inputs = Vec::with_capacity(bands);
    let mut band0_plan = None;
    for (band, (lo, hi)) in hash_bands(bands).into_iter().enumerate() {
        let band_df = df
            .clone()
            .filter(
                col(HASH_LABEL)
                    .gt_eq(lit(lo))
                    .and(col(HASH_LABEL).lt_eq(lit(hi))),
            )?
            .select_columns(columns)?
            // planning-only: proves the scan partitions hash-ordered; fold_band merges, not the SPM
            .sort(vec![col(HASH_LABEL).sort(true, false)])?;
        let task_ctx = Arc::new(band_df.task_ctx());
        let plan = band_df.create_physical_plan().await?;
        let Some(streams) = band_streams(plan.clone(), task_ctx)? else {
            log::info!(
                "[trace_id: {trace_id}] [PromQL] streaming fused agg fallback: band {band} plan cannot stream in order"
            );
            return Ok(None);
        };
        band0_plan.get_or_insert(plan);
        band_inputs.push(streams);
    }
    let band0_plan = band0_plan.expect("target_partitions is at least one band");
    Ok(Some((band_inputs, band0_plan)))
}

/// The hash-ordered inputs a band folds from: the merge's own child
/// partitions, so the row-level merge node itself is never executed.
fn band_streams(
    plan: Arc<dyn ExecutionPlan>,
    task_ctx: Arc<TaskContext>,
) -> Result<Option<Vec<SendableRecordBatchStream>>> {
    if plan_contains_sort(&plan) {
        return Ok(None);
    }
    if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() {
        return Ok(Some(execute_stream_partitioned(
            merge.input().clone(),
            task_ctx,
        )?));
    }
    if plan.properties().output_partitioning().partition_count() == 1 && hash_ordered(&plan) {
        return Ok(Some(vec![execute_stream(plan, task_ctx)?]));
    }
    Ok(None)
}

fn plan_contains_sort(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<SortExec>().is_some()
        || plan
            .children()
            .iter()
            .any(|child| plan_contains_sort(child))
}

/// Aborts the band tasks when `timeout` elapses before they all finish.
async fn run_bands(
    band_inputs: Vec<Vec<SendableRecordBatchStream>>,
    params: Arc<FoldParams>,
    timeout: u64,
) -> Result<Vec<(GroupAccs, usize)>> {
    let mut tasks = Vec::with_capacity(band_inputs.len());
    let mut abort_handles = Vec::with_capacity(band_inputs.len());
    for streams in band_inputs {
        let task = tokio::task::spawn(fold_band(streams, params.clone()));
        abort_handles.push(task.abort_handle());
        tasks.push(task);
    }
    tokio::select! {
        joined = futures::future::try_join_all(tasks) => {
            joined
                .map_err(|e| DataFusionError::Execution(e.to_string()))?
                .into_iter()
                .collect()
        }
        _ = tokio::time::sleep(Duration::from_secs(timeout)) => {
            for handle in abort_handles {
                handle.abort();
            }
            Err(DataFusionError::Plan(
                Error::ErrorCode(ErrorCodes::SearchTimeout(
                    "[PromQL] streaming fused agg timeout".to_string(),
                ))
                .to_string(),
            ))
        }
    }
}

fn hash_ordered(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.properties().output_ordering().is_some_and(|ordering| {
        let sort = ordering.first();
        !sort.options.descending
            && sort
                .expr
                .downcast_ref::<Column>()
                .is_some_and(|column| column.name() == HASH_LABEL)
    })
}

/// Merges the band's chains one series at a time: a series is one run per
/// chain, so the merge costs one round of cursor checks per series instead of
/// one heap operation per row.
async fn fold_band(
    streams: Vec<SendableRecordBatchStream>,
    params: Arc<FoldParams>,
) -> Result<(GroupAccs, usize)> {
    let mut groups = GroupAccs::new();
    let mut cursors = Vec::with_capacity(streams.len());
    for stream in streams {
        cursors.push(ChainCursor::start(stream).await?);
    }
    let mut samples: Vec<Sample> = Vec::new();
    let mut series_count = 0;
    while let Some(hash) = cursors.iter().filter_map(ChainCursor::head_hash).min() {
        samples.clear();
        let mut key = None;
        for cursor in &mut cursors {
            if cursor.head_hash() != Some(hash) {
                continue;
            }
            if key.is_none() {
                let batch = cursor.batch.as_ref().expect("head_hash implies a batch");
                key = Some(group_key_at(batch, cursor.row, &params, &groups)?);
            }
            cursor
                .consume_run(hash, params.offset, &mut samples)
                .await?;
        }
        let (sig, labels) = key.expect("the minimum head hash has a contributing chain");
        // classic parity: chains interleave in time, so restore per-series order
        samples.sort_unstable_by_key(|sample| sample.timestamp);
        let entry = match groups.entry(sig) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(GroupEntry {
                labels: labels.unwrap_or_default(),
                acc: FusedAccumulator::new(params.op, params.timestamps.len()),
            }),
        };
        fold_series(
            &mut entry.acc,
            &samples,
            params.range,
            params.func.as_ref(),
            params.counter_kind,
            &params.eval_ctx,
            &params.timestamps,
        );
        series_count += 1;
    }
    Ok((groups, series_count))
}

fn group_key_at(
    batch: &RecordBatch,
    row: usize,
    params: &FoldParams,
    groups: &GroupAccs,
) -> Result<(u64, Option<Labels>)> {
    let label_cols = params
        .group_cols
        .iter()
        .map(|name| {
            LabelColumn::try_from_array(batch[name.as_str()].as_ref()).ok_or_else(|| {
                DataFusionError::Execution(format!("label column {name} is not Utf8 or Utf8View"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let sig = group_signature(&label_cols, &params.group_cols, row);
    let labels = (!groups.contains_key(&sig))
        .then(|| materialize_labels(&label_cols, &params.group_cols, row));
    Ok((sig, labels))
}

fn group_signature(cols: &[LabelColumn<'_>], names: &[String], row: usize) -> u64 {
    let mut hasher = gxhash::new_hasher();
    for (values, name) in cols.iter().zip(names) {
        if !values.is_null(row) {
            hasher.write(name.as_bytes());
            hasher.write(values.value(row).as_bytes());
        }
    }
    hasher.finish()
}

fn materialize_labels(cols: &[LabelColumn<'_>], names: &[String], row: usize) -> Labels {
    cols.iter()
        .zip(names)
        .filter(|(values, _)| !values.is_null(row))
        .map(|(values, name)| Arc::new(Label::new(name.as_str(), values.value(row))))
        .collect()
}

/// Merges the band-local groups in band order and materializes the result;
/// groups whose accumulator produced no samples are dropped like the
/// materializing path drops no-output series.
fn merge_folds(folds: Vec<GroupAccs>, timestamps: &[i64]) -> Value {
    let mut folds = folds.into_iter();
    let Some(mut merged) = folds.next() else {
        return Value::None;
    };
    for fold in folds {
        for (sig, entry) in fold {
            match merged.entry(sig) {
                Entry::Occupied(mut occupied) => occupied.get_mut().acc.merge(entry.acc),
                Entry::Vacant(vacant) => {
                    vacant.insert(entry);
                }
            }
        }
    }
    let results: Vec<RangeValue> = merged
        .into_values()
        .filter_map(|entry| {
            let samples = entry.acc.into_samples(timestamps);
            if samples.is_empty() {
                return None;
            }
            Some(RangeValue {
                labels: entry.labels,
                samples,
                exemplars: None,
                time_window: None,
            })
        })
        .collect();
    if results.is_empty() {
        Value::None
    } else {
        Value::Matrix(results)
    }
}

#[cfg(test)]
mod tests {
    use config::meta::promql::value::TimeWindow;
    use datafusion::{
        arrow::array::{Float64Array, Int64Array, StringArray, UInt64Array},
        datasource::MemTable,
        logical_expr::SortExpr,
        prelude::SessionConfig,
    };
    use itertools::Itertools;
    use promql_parser::label::Labels as ModifierLabels;

    use super::*;
    use crate::{functions, fused::eval::fused_range_agg};

    type CanonicalSeries = (Vec<(String, String)>, Vec<(i64, u64)>);

    const SECOND: i64 = 1_000_000;
    const BASE: i64 = 1_000 * SECOND;

    fn eval_ctx() -> EvalContext {
        EvalContext::new(
            BASE + 60 * SECOND,
            BASE + 180 * SECOND,
            60 * SECOND,
            "test".into(),
        )
    }

    fn arrow_schema() -> Arc<Schema> {
        use datafusion::arrow::datatypes::Field;
        Arc::new(Schema::new(vec![
            Field::new(TIMESTAMP_COL_NAME, DataType::Int64, false),
            Field::new(HASH_LABEL, DataType::UInt64, false),
            Field::new(VALUE_LABEL, DataType::Float64, false),
            Field::new("instance", DataType::Utf8, true),
            Field::new("path", DataType::Utf8, true),
        ]))
    }

    /// (hash, seconds offset, value, instance, path)
    type Row = (u64, i64, f64, Option<&'static str>, Option<&'static str>);

    /// The test series: counters, a counter reset, a late-only series, and a
    /// series with a null label, spread over the full hash space.
    fn test_rows() -> Vec<Row> {
        let dense = [10, 50, 70, 110, 130, 170];
        let series = |hash, values: [f64; 6], instance, path| {
            dense
                .into_iter()
                .zip(values)
                .map(move |(ts, value)| (hash, ts, value, instance, path))
        };
        let mut rows: Vec<Row> = series(
            100,
            [0.1, 40.7, 45.2, 85.9, 90.4, 130.8],
            Some("a"),
            Some("/one"),
        )
        .chain(series(
            200,
            [0.3, 80.1, 90.6, 170.2, 180.9, 260.5],
            Some("b"),
            Some("/one"),
        ))
        // counter reset (25.1 -> 3.4) crossing the partition split below
        .chain(series(
            5,
            [0.2, 20.4, 25.1, 3.4, 50.3, 70.9],
            Some("c"),
            Some("/two"),
        ))
        .collect();
        // samples only in the last window
        rows.push((u64::MAX - 3, 130, 7.5, Some("z"), Some("/two")));
        rows.push((u64::MAX - 3, 170, 11.25, Some("z"), Some("/two")));
        // null instance label
        rows.push((42, 50, 1.0, None, Some("/two")));
        rows.push((42, 110, 3.0, None, Some("/two")));
        rows
    }

    fn rows_to_batch(mut rows: Vec<Row>) -> RecordBatch {
        rows.sort_by_key(|row| (row.0, row.1));
        RecordBatch::try_new(
            arrow_schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|row| BASE + row.1 * SECOND),
                )),
                Arc::new(UInt64Array::from_iter_values(rows.iter().map(|row| row.0))),
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.2))),
                Arc::new(StringArray::from(
                    rows.iter().map(|row| row.3).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|row| row.4).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    /// Two overlapping sorted "files": every series with more than one sample
    /// is split across both, so only the ordered merge sees it whole.
    fn sorted_partitions() -> Vec<Vec<RecordBatch>> {
        let (even, odd): (Vec<Row>, Vec<Row>) =
            test_rows()
                .into_iter()
                .enumerate()
                .partition_map(|(index, row)| {
                    if index.is_multiple_of(2) {
                        itertools::Either::Left(row)
                    } else {
                        itertools::Either::Right(row)
                    }
                });
        vec![vec![rows_to_batch(even)], vec![rows_to_batch(odd)]]
    }

    fn session_ctx() -> SessionContext {
        let mut config = SessionConfig::new().with_target_partitions(3);
        config.options_mut().optimizer.prefer_existing_sort = true;
        SessionContext::new_with_config(config)
    }

    fn register_sorted_table(ctx: &SessionContext) {
        let sort_order: Vec<SortExpr> = vec![
            col(HASH_LABEL).sort(true, false),
            col(TIMESTAMP_COL_NAME).sort(true, false),
        ];
        let table = MemTable::try_new(arrow_schema(), sorted_partitions())
            .unwrap()
            .with_sort_order(vec![sort_order]);
        ctx.register_table(format!("m{STREAMING_AGG_TABLE_SUFFIX}"), Arc::new(table))
            .unwrap();
    }

    /// The same data as a materialized matrix for the reference evaluator.
    fn reference_matrix(range: Duration) -> Vec<RangeValue> {
        let mut by_hash: HashMap<u64, RangeValue> = HashMap::new();
        for (hash, ts, value, instance, path) in test_rows() {
            let entry = by_hash.entry(hash).or_insert_with(|| RangeValue {
                labels: [("instance", instance), ("path", path)]
                    .into_iter()
                    .filter_map(|(name, value)| Some(Arc::new(Label::new(name, value?))))
                    .collect(),
                samples: vec![],
                exemplars: None,
                time_window: Some(TimeWindow::new(range)),
            });
            entry.samples.push(Sample::new(BASE + ts * SECOND, value));
        }
        let mut matrix: Vec<RangeValue> = by_hash.into_values().collect();
        for series in &mut matrix {
            series
                .samples
                .sort_unstable_by_key(|sample| sample.timestamp);
        }
        matrix
    }

    fn canonical_matrix(value: Value) -> Vec<CanonicalSeries> {
        let matrix = match value {
            Value::Matrix(matrix) => matrix,
            Value::None => return vec![],
            value => panic!("expected matrix or none, got {}", value.get_type()),
        };
        let mut canonical = matrix
            .into_iter()
            .map(|series| {
                let mut labels = series
                    .labels
                    .iter()
                    .map(|label| (label.name.clone(), label.value.clone()))
                    .collect::<Vec<_>>();
                labels.sort();
                let samples = series
                    .samples
                    .iter()
                    .map(|sample| (sample.timestamp, sample.value.to_bits()))
                    .collect::<Vec<_>>();
                (labels, samples)
            })
            .collect::<Vec<_>>();
        canonical.sort_by(|a, b| a.0.cmp(&b.0));
        canonical
    }

    /// Labels and timestamps must match exactly; values may drift in the last
    /// bits because streaming folds series in hash order, not matrix order.
    fn assert_matrix_close(
        expected: Vec<CanonicalSeries>,
        actual: Vec<CanonicalSeries>,
        context: &str,
    ) {
        assert_eq!(expected.len(), actual.len(), "{context}: series count");
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_eq!(expected.0, actual.0, "{context}: labels");
            assert_eq!(expected.1.len(), actual.1.len(), "{context}: sample count");
            for (&(expected_ts, expected_bits), &(actual_ts, actual_bits)) in
                expected.1.iter().zip(&actual.1)
            {
                assert_eq!(expected_ts, actual_ts, "{context}: timestamps");
                if expected_bits == actual_bits {
                    continue;
                }
                let expected_value = f64::from_bits(expected_bits);
                let actual_value = f64::from_bits(actual_bits);
                assert!(
                    expected_value.is_finite() && actual_value.is_finite(),
                    "{context}: non-finite values must match exactly"
                );
                let tolerance = expected_value.abs().max(actual_value.abs()) * 1e-12;
                assert!(
                    (expected_value - actual_value).abs() <= tolerance,
                    "{context}: {expected_value} vs {actual_value}"
                );
            }
        }
    }

    fn by(labels: &[&str]) -> Option<LabelModifier> {
        Some(LabelModifier::Include(ModifierLabels {
            labels: labels.iter().map(|label| label.to_string()).collect(),
        }))
    }

    async fn run_streaming(
        ctx: &SessionContext,
        modifier: &Option<LabelModifier>,
        func_name: &str,
        op: FusedAggOp,
        range: Duration,
    ) -> Option<Value> {
        let func: Arc<dyn RangeFunc> = Arc::from(functions::fusable_range_func(func_name).unwrap());
        let eval_ctx = eval_ctx();
        streaming_fused_agg(
            ctx,
            &arrow_schema(),
            StreamingSelector {
                table_name: "m",
                matchers: &Matchers::empty(),
                start: eval_ctx.start,
                end: eval_ctx.end,
                step: eval_ctx.step,
                lookback: crate::micros(range),
                offset: 0,
            },
            FusedShape { op, func, range },
            modifier,
            &eval_ctx,
            10,
        )
        .await
        .unwrap()
    }

    #[test]
    fn test_hash_bands_cover_the_full_space_contiguously() {
        for count in [1, 3, 7, 16] {
            let bands = hash_bands(count);
            assert_eq!(bands.len(), count);
            assert_eq!(bands[0].0, 0);
            assert_eq!(bands[count - 1].1, u64::MAX);
            for pair in bands.windows(2) {
                assert_eq!(pair[0].1.wrapping_add(1), pair[1].0);
            }
        }
    }

    #[test]
    fn test_group_label_columns_resolution() {
        let schema = arrow_schema();
        assert_eq!(group_label_columns(&None, &schema, "rate"), Some(vec![]));
        assert_eq!(
            group_label_columns(&by(&["path", "instance", "path"]), &schema, "rate"),
            Some(vec!["instance".to_string(), "path".to_string()])
        );
        // absent columns group like an absent label: no column to read
        assert_eq!(
            group_label_columns(&by(&["nope", HASH_LABEL, VALUE_LABEL]), &schema, "rate"),
            Some(vec![])
        );
        // rate strips the metric name; last_over_time keeps it (not in schema here)
        assert_eq!(
            group_label_columns(&by(&[NAME_LABEL]), &schema, "rate"),
            Some(vec![])
        );
        let without = Some(LabelModifier::Exclude(ModifierLabels {
            labels: vec!["instance".to_string()],
        }));
        assert_eq!(group_label_columns(&without, &schema, "rate"), None);
    }

    #[tokio::test]
    async fn test_streaming_matches_fused_for_all_pairs() {
        let ctx = session_ctx();
        register_sorted_table(&ctx);
        let range = Duration::from_secs(60);
        let eval_ctx = eval_ctx();

        let agg_cases = [
            FusedAggOp::Avg,
            FusedAggOp::Count,
            FusedAggOp::Group,
            FusedAggOp::Max,
            FusedAggOp::Min,
            FusedAggOp::Stddev,
            FusedAggOp::Stdvar,
            FusedAggOp::Sum,
        ];
        let func_cases = ["rate", "increase", "sum_over_time", "last_over_time"];
        let modifiers = [
            None,
            by(&["path"]),
            by(&["instance", "path"]),
            by(&["nope"]),
        ];

        for op in agg_cases {
            for func_name in func_cases {
                for modifier in &modifiers {
                    let func = functions::fusable_range_func(func_name).unwrap();
                    let expected = fused_range_agg(
                        modifier,
                        Value::Matrix(reference_matrix(range)),
                        func.as_ref(),
                        op,
                        &eval_ctx,
                    )
                    .unwrap();

                    let actual = run_streaming(&ctx, modifier, func_name, op, range)
                        .await
                        .expect("streaming path must not fall back on the sorted table");

                    assert_matrix_close(
                        canonical_matrix(expected),
                        canonical_matrix(actual),
                        &format!(
                            "streaming {}({func_name}) (modifier: {modifier:?})",
                            op.name()
                        ),
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_streaming_falls_back_without_sorted_table() {
        let ctx = session_ctx();
        let table = MemTable::try_new(arrow_schema(), sorted_partitions()).unwrap();
        ctx.register_table("m", Arc::new(table)).unwrap();

        let result = run_streaming(
            &ctx,
            &None,
            "rate",
            FusedAggOp::Sum,
            Duration::from_secs(60),
        )
        .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_streaming_falls_back_when_ordering_is_not_declared() {
        let ctx = session_ctx();
        // same data registered under the sorted name but without the ordering
        // declaration: the plan needs a real sort, so the gate must reject it
        let table = MemTable::try_new(arrow_schema(), sorted_partitions()).unwrap();
        ctx.register_table(format!("m{STREAMING_AGG_TABLE_SUFFIX}"), Arc::new(table))
            .unwrap();

        let result = run_streaming(
            &ctx,
            &None,
            "rate",
            FusedAggOp::Sum,
            Duration::from_secs(60),
        )
        .await;
        assert!(result.is_none());
    }
}
