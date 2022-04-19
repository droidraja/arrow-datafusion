// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines the projection execution plan. A projection determines which columns or expressions
//! are returned from a query. The SQL statement `SELECT a, b, a+b FROM t1` is an example
//! of a projection on table `t1` where the expressions `a`, `b`, and `a+b` are the
//! projection expressions. `SELECT` without `FROM` will only evaluate expressions.

use arrow::compute::concat;
use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::error::{DataFusionError, Result};
use crate::physical_plan::{DisplayFormatType, ExecutionPlan, Partitioning};
use arrow::datatypes::{Schema, SchemaRef};
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::record_batch::RecordBatch;

use super::expressions::PhysicalSortExpr;
use super::{RecordBatchStream, SendableRecordBatchStream, Statistics};
use crate::execution::context::TaskContext;
use async_trait::async_trait;
use datafusion_common::OuterQueryCursor;
use futures::stream::Stream;
use futures::stream::StreamExt;

/// Execution plan for a sub query
#[derive(Debug)]
pub struct SubQueryExec {
    /// Sub queries
    sub_queries: Vec<Arc<dyn ExecutionPlan>>,
    /// Merged schema
    schema: SchemaRef,
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
    /// Cursor used to send outer query column values to sub queries
    cursor: Arc<OuterQueryCursor>,
}

impl SubQueryExec {
    /// Create a projection on an input
    pub fn try_new(
        sub_queries: Vec<Arc<dyn ExecutionPlan>>,
        input: Arc<dyn ExecutionPlan>,
        cursor: Arc<OuterQueryCursor>,
    ) -> Result<Self> {
        let input_schema = (*input.schema()).clone();

        let merged_schema = Schema::try_merge(
            vec![input_schema].into_iter().chain(
                sub_queries
                    .iter()
                    .map(|s| (*s.schema()).clone())
                    .collect::<Vec<_>>(),
            ),
        )?;

        if merged_schema.fields().len()
            != input.schema().fields().len() + sub_queries.len()
        {
            return Err(DataFusionError::Plan("One or more correlated sub queries use same column names which is not supported".to_string()));
        }

        Ok(Self {
            sub_queries,
            schema: Arc::new(merged_schema),
            input,
            cursor,
        })
    }
}

#[async_trait]
impl ExecutionPlan for SubQueryExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        let mut res = vec![self.input.clone()];
        res.extend(self.sub_queries.iter().cloned());
        res
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        self.input.output_partitioning()
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        self.input.output_ordering()
    }

    fn maintains_input_order(&self) -> bool {
        // tell optimizer this operator doesn't reorder its input
        true
    }

    fn relies_on_input_order(&self) -> bool {
        true
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() < 2 {
            return Err(DataFusionError::Internal(format!(
                "SubQueryExec should have not less than 2 children but got {}",
                children.len()
            )));
        }

        Ok(Arc::new(SubQueryExec::try_new(
            children.iter().skip(1).cloned().collect(),
            children[0].clone(),
            self.cursor.clone(),
        )?))
    }

    async fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let stream = self.input.execute(partition, context.clone()).await?;
        let cursor = self.cursor.clone();
        let sub_queries = self.sub_queries.clone();
        let context = context.clone();
        let size_hint = stream.size_hint();
        let schema = self.schema.clone();
        let res_stream = stream.then(move |batch| {
            let cursor = cursor.clone();
            let context = context.clone();
            let sub_queries = sub_queries.clone();
            let schema = schema.clone();
            async move {
                let batch = batch?;
                let b = Arc::new(batch.clone());
                cursor.set_batch(b)?;
                let mut sub_query_arrays = vec![Vec::new(); sub_queries.len()];
                for i in 0..batch.num_rows() {
                    cursor.set_position(i)?;
                    for (sub_query_i, sub_query) in sub_queries.iter().enumerate() {
                        if sub_query.output_partitioning().partition_count() != 1 {
                            return Err(ArrowError::ComputeError(format!("Sub query should have only one partition but got {}", sub_query.output_partitioning().partition_count())))
                        }
                        let mut stream = sub_query.execute(0, context.clone()).await?;
                        let res = stream.next().await;
                        if let Some(sub_query_batch) = res {
                            let sub_query_batch = sub_query_batch?;
                            if sub_query_batch.column(0).len() != 1 {
                                return Err(ArrowError::ComputeError("Sub query should return exactly one row".to_string()))
                            } else {
                                sub_query_arrays[sub_query_i].push(sub_query_batch.column(0).clone());
                            }
                        } else {
                            return Err(ArrowError::ComputeError("Sub query returned empty result set but exactly one row is expected".to_string()))
                        }
                    }
                }
                let mut new_columns = batch.columns().to_vec();
                for sub_query_array in sub_query_arrays {
                    new_columns.push(concat(sub_query_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>().as_slice())?);
                }
                Ok(RecordBatch::try_new(schema.clone(), new_columns)?)
            }
        });
        Ok(Box::pin(SubQueryStream {
            schema: self.schema.clone(),
            stream: Box::pin(res_stream),
            size_hint,
        }))
    }

    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "SubQueryExec")
    }

    fn statistics(&self) -> Statistics {
        // TODO
        self.input.statistics()
    }
}

/// SubQuery iterator
struct SubQueryStream {
    stream: Pin<Box<dyn Stream<Item = ArrowResult<RecordBatch>> + Send>>,
    schema: SchemaRef,
    size_hint: (usize, Option<usize>),
}

impl Stream for SubQueryStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.stream.poll_next_unpin(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.size_hint.clone()
    }
}

impl RecordBatchStream for SubQueryStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
