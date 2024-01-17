use crate::driver::safe::{CudaDevice, DriverError};
use crate::driver::{result::graph, sys};
use std::sync::Arc;

pub struct Graph {
    _graph: sys::CUgraph,
    instance: sys::CUgraphExec,
    device: Arc<CudaDevice>,
}
pub struct Handle {
    // stream: sys::CUstream,
    device: Arc<CudaDevice>,
}

impl Graph {
    pub fn start_capture(device: &Arc<CudaDevice>) -> Result<Handle, DriverError> {
        Handle::new(device)
    }

    pub fn launch(&self) -> Result<(), DriverError> {
        graph::launch(self.instance, self.device.stream)?;
        Ok(())
    }
}

impl Handle {
    pub fn new(device: &Arc<CudaDevice>) -> Result<Self, DriverError> {
        let stream = device.stream;
        // let status = result::stream_is_capturing(stream)?;
        let mode = sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL;
        graph::stream_begin_capture_v2(stream, mode)?;
        Ok(Self {
            device: device.clone(),
        })
    }

    pub fn end_capture(&self) -> Result<Graph, DriverError> {
        let (_graph, instance) = graph::stream_end_capture(self.device.stream)?;
        Ok(Graph {
            _graph,
            instance,
            device: self.device.clone(),
        })
    }
}
