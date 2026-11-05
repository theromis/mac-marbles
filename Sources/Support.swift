import Metal
import MetalFX

enum MTLFXSupport {
    static func denoiserAvailable(on device: MTLDevice) -> Bool {
        MTLFXTemporalDenoisedScalerDescriptor.supportsDevice(device)
    }
}
