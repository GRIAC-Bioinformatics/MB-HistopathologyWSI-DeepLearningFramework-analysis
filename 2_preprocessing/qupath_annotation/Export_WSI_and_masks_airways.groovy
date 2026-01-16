import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// Get the main QuPath data structures
def imageData = getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

// Define downsample value for export resolution & output directory, creating directory if necessary
def downsample = 20.0
def pathOutput = buildFilePath(QPEx.PROJECT_BASE_DIR, 'WSI_and_masks_area')
mkdirs(pathOutput)

// Export WSI
def requestFull = RegionRequest.createInstance(server, downsample)
String wsiName = String.format('%s',
            server.getMetadata().getName()
    )
writeImageRegion(server, requestFull, pathOutput + '/' + wsiName + '_WSI_downscaled.png')



// Export each airway
def region = RegionRequest.createInstance(server, downsample)
def img = server.readBufferedImage(region)
def areas = hierarchy.getAnnotationObjects().findAll{it.getPathClass() == getPathClass("Area")}

areas.each{
    def imgMask = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    def g2d = imgMask.createGraphics()
    
    g2d.setColor(Color.WHITE)
    g2d.scale(1.0/downsample, 1.0/downsample)
    g2d.translate(-region.getX(), -region.getY())

    def roi = it.getROI()
    def region_airway = RegionRequest.createInstance(server.getPath(), downsample, roi)
    def shape = RoiTools.getShape(roi)
    g2d.fill(shape)
    
    String name = String.format('%s__Area_(%d,%d)__',
            wsiName,
            region_airway.getWidth(),
            region_airway.getHeight()
    )
    
    // Export the mask
    def fileMask = new File(pathOutput, name + 'mask.png')
    ImageIO.write(imgMask, 'PNG', fileMask)
    
    }