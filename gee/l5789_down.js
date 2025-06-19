// author: xin luo
// create: 2020.11.9
// des: This code export the selected l578 image to the google drive.

/////////////////////////////////////////
var image_id = 'LANDSAT/LC09/C02/T1_L2/LC09_183051_20220922'


var region = ee.Geometry.Rectangle(17.29, 12.47, 17.75, 12.91)
var bands_sel = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'] // landsat 9
// var bands_sel = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'] // landsat 8
var bands_vis = ['SR_B5', 'SR_B4', 'SR_B3']   // landsat 8
// var bands_sel = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']    // landsat 5,7
// var bands_vis = ['B4', 'B3', 'B2']                   // landsat 5,7

// source images and scene region


var l5789_scene = ee.Image(image_id).clip(region).select(bands_sel)
print('l5789_scene', l5789_scene)


// visualization
Map.centerObject(region, 9);
Map.addLayer(l5789_scene, {bands: bands_vis, min:0, max:3000}, 'l5789_scene');


// // Export to Google Drive
// Export.image.toDrive({
//   image: l5789_scene,
//   description: 'l8_scene_30',
//   folder: 'landsat5789_water',
//   scale: 30,
//   fileFormat: 'GeoTIFF',
//   region: region
//   });



