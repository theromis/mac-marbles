import AppKit
import CoreText
import Metal

/// Draws a made-up newspaper page, so the refraction of each marble can be judged against print.
/// All of the text is invented for this demo. Type is large because each marble is small on screen.
func makeNewspaperTexture(device: MTLDevice, queue: MTLCommandQueue) -> MTLTexture {
    let width = 2048
    let height = 1424
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
    context.setFillColor(gray: 1, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))

    func draw(_ text: String, font: String, size: CGFloat, in rect: CGRect, centered: Bool = false) {
        let style = NSMutableParagraphStyle()
        style.alignment = centered ? .center : .justified
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont(name: font, size: size) ?? NSFont.systemFont(ofSize: size),
            .foregroundColor: NSColor.black,
            .paragraphStyle: style,
        ]
        let framesetter = CTFramesetterCreateWithAttributedString(NSAttributedString(string: text, attributes: attributes))
        let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: 0, length: 0), CGPath(rect: rect, transform: nil), nil)
        CTFrameDraw(frame, context)
    }
    func rule(y: CGFloat, thickness: CGFloat) {
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 60, y: y, width: CGFloat(width) - 120, height: thickness))
    }

    let h = CGFloat(height)
    draw("THE GLASS GAZETTE", font: "Times New Roman Bold", size: 190, in: CGRect(x: 60, y: h - 245, width: 1928, height: 225), centered: true)
    rule(y: h - 262, thickness: 10)
    draw("MARBLES ROLL DOWNHILL AGAIN", font: "Times New Roman Bold", size: 108, in: CGRect(x: 60, y: h - 410, width: 1928, height: 135), centered: true)
    rule(y: h - 422, thickness: 5)

    let story = """
    A round of glass entered the wooden channel this morning and left it some twenty seconds later, witnesses say. \
    Light passing through it was bent twice, once on the way in and once on the way out. Readers who look closely \
    will find these very words upside down inside every marble that rolls across the page. Experts call this \
    refraction. The marbles had no comment. Officials confirm the next one is already on its way down the spiral, \
    and that the drain near the column remains open to all.
    """
    let body = String(repeating: story + " ", count: 3)
    let columns = 4
    let gap: CGFloat = 44
    let columnWidth = (CGFloat(width) - 120 - gap * CGFloat(columns - 1)) / CGFloat(columns)
    for c in 0..<columns {
        let x = 60 + CGFloat(c) * (columnWidth + gap)
        // A different starting point per column, so the page does not look tiled.
        let start = body.index(body.startIndex, offsetBy: c * 137)
        draw(String(body[start...]), font: "Times New Roman", size: 44, in: CGRect(x: x, y: 50, width: columnWidth, height: h - 490))
        if c > 0 {
            context.setFillColor(gray: 0, alpha: 1)
            context.fill(CGRect(x: x - gap / 2 - 1, y: 50, width: 3, height: h - 490))
        }
    }

    let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: width, height: height, mipmapped: true)
    descriptor.usage = .shaderRead
    descriptor.storageMode = .shared
    let texture = device.makeTexture(descriptor: descriptor)!
    texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                    withBytes: context.data!, bytesPerRow: context.bytesPerRow)
    let commands = queue.makeCommandBuffer()!
    let blit = commands.makeBlitCommandEncoder()!
    blit.generateMipmaps(for: texture)
    blit.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return texture
}
