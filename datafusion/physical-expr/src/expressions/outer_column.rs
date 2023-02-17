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

//! Column expression

use std::any::Any;
use std::sync::Arc;

use arrow::{
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};

use crate::physical_expr::down_cast_any_ref;
use crate::PhysicalExpr;
use datafusion_common::OuterQueryCursor;
use datafusion_common::Result;
use datafusion_expr::ColumnarValue;

/// Represents the column at a given index in a RecordBatch
#[derive(Debug, Clone)]
pub struct OuterColumn {
    name: String,
    outer_query_cursor: Arc<OuterQueryCursor>,
}

impl OuterColumn {
    /// Create a new column expression
    pub fn new(name: &str, outer_query_cursor: Arc<OuterQueryCursor>) -> Self {
        Self {
            name: name.to_owned(),
            outer_query_cursor,
        }
    }
}

impl std::fmt::Display for OuterColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl PhysicalExpr for OuterColumn {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Get the data type of this expression, given the schema of the input
    fn data_type(&self, _: &Schema) -> Result<DataType> {
        Ok(self
            .outer_query_cursor
            .schema()
            .field_with_name(self.name.as_str())?
            .data_type()
            .clone())
    }

    /// Decide whehter this expression is nullable, given the schema of the input
    fn nullable(&self, _: &Schema) -> Result<bool> {
        Ok(self
            .outer_query_cursor
            .schema()
            .field_with_name(self.name.as_str())?
            .is_nullable())
    }

    /// Evaluate the expression
    fn evaluate(&self, _: &RecordBatch) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Scalar(
            self.outer_query_cursor.field_value(self.name.as_str())?,
        ))
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(self)
    }
}

impl PartialEq<dyn Any> for OuterColumn {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| {
                self.name == x.name && self.outer_query_cursor == x.outer_query_cursor
            })
            .unwrap_or(false)
    }
}
