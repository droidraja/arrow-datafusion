use crate::Result;
use crate::{DataFusionError, ScalarValue};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

pub struct OuterQueryCursor {
    schema: SchemaRef,
    record_batch_and_pos: std::sync::RwLock<Option<(Arc<RecordBatch>, usize)>>,
}

impl Debug for OuterQueryCursor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OuterQueryCursor").finish()
    }
}

impl OuterQueryCursor {
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            record_batch_and_pos: std::sync::RwLock::new(None),
        }
    }

    pub fn set_batch(&self, record_batch: Arc<RecordBatch>) -> Result<()> {
        let mut option = self
            .record_batch_and_pos
            .write()
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        *option = Some((record_batch, 0));
        Ok(())
    }

    pub fn set_position(&self, pos: usize) -> Result<()> {
        if let Some(mut batch_and_pos) = self
            .record_batch_and_pos
            .write()
            .map_err(|e| DataFusionError::Execution(e.to_string()))?
            .as_mut()
        {
            batch_and_pos.1 = pos;
            Ok(())
        } else {
            Err(DataFusionError::Execution(
                "Trying to set position on uninitialized OuterQueryCursor".to_string(),
            ))
        }
    }

    pub fn field_value(&self, name: &str) -> Result<ScalarValue> {
        let guard = self
            .record_batch_and_pos
            .read()
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        let (batch, pos) = guard.as_ref().ok_or_else(|| {
            DataFusionError::Execution(format!(
                "Trying to get '{}' column on uninitialized OuterQueryCursor",
                name
            ))
        })?;

        let col_index = batch.schema().index_of(name)?;
        ScalarValue::try_from_array(batch.column(col_index), *pos)
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
