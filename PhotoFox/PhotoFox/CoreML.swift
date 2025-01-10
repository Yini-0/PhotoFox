import UIKit
import Vision
import CoreML

class CoreMLService {
    static let singleton = CoreMLService()
    
    private var model: MobileNetV2FP16?
    
    private init() {}
    
    private func loadModel() async throws -> MobileNetV2FP16 {
        let config = MLModelConfiguration()
        return try MobileNetV2FP16(configuration: config)
    }
    
    func getModel() async throws -> MobileNetV2FP16 {
        if let model = model {
            return model
        } else {
            let loadedModel = try await loadModel()
            self.model = loadedModel
            return loadedModel
        }
    }
    
    func predict(image: UIImage) async throws -> ImagePrediction {
        try Task.checkCancellation()
        
        let model = try await getModel()
        
        guard let input = try prepareImageInput(image: image) else { return ImagePrediction(label: "", confidence: 0.0) }
        
        let output = try model.prediction(image: input)
        
        print("Prediction: \(output.classLabelProbs)")
        
        return ImagePrediction(
            label: output.classLabel,
            confidence: output.classLabelProbs[output.classLabel] ?? 0.0
        )
    }
    
    func detectFaces(in image: UIImage) {
        guard let cgImage = image.cgImage else { return }

        // Create a face detection request
        let request = VNDetectFaceRectanglesRequest { request, error in
            if let error = error {
                print("Face detection error: \(error.localizedDescription)")
                return
            }

            guard let results = request.results as? [VNFaceObservation] else { return }

            for faceObservation in results {
                print("Detected face at \(faceObservation.boundingBox)")
            }
        }

        // Create a request handler
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            // Perform the face detection request
            try handler.perform([request])
        } catch {
            print("Failed to perform face detection: \(error.localizedDescription)")
        }
    }
}

struct ImagePrediction {
    var label: String
    var confidence: Double
}

func prepareImageInput(image: UIImage) throws -> CVPixelBuffer? {
    let newSize = CGSize(width: 224.0, height: 224.0)
    UIGraphicsBeginImageContext(newSize)
    image.draw(in: CGRect(origin: CGPoint.zero, size: newSize))
    
    guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
        return nil
    }
    
    UIGraphicsEndImageContext()
    
    // convert to pixel buffer
    let attributes = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
              kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                     Int(newSize.width),
                                     Int(newSize.height),
                                     kCVPixelFormatType_32ARGB,
                                     attributes,
                                     &pixelBuffer)
    
    guard let createdPixelBuffer = pixelBuffer, status == kCVReturnSuccess else {
        return nil
    }
    
    CVPixelBufferLockBaseAddress(createdPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    let pixelData = CVPixelBufferGetBaseAddress(createdPixelBuffer)
    
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(data: pixelData,
                                  width: Int(newSize.width),
                                  height: Int(newSize.height),
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(createdPixelBuffer),
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
        return nil
    }
    
    context.translateBy(x: 0, y: newSize.height)
    context.scaleBy(x: 1.0, y: -1.0)
    
    UIGraphicsPushContext(context)
    resizedImage.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
    UIGraphicsPopContext()
    CVPixelBufferUnlockBaseAddress(createdPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    
    return createdPixelBuffer
}

extension UIImage {
    func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        guard let cgImage = self.cgImage, let ctx = context else {
            return nil
        }
        
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}
