def name = getProjectEntry().getImageName()
def imageData = getCurrentImageData()
def server = getCurrentServer()
def metadata = server.getMetadata()
if (metadata.getName() == name) {
    println 'Name already correct for ' + name
    return
}
println 'Updating name from ' + metadata.getName() + ' to ' + name
def metadata2 = new qupath.lib.images.servers.ImageServerMetadata.Builder(metadata).name(name).build()
imageData.updateServerMetadata(metadata2)