#1: Get a TAU profile with comm matrix data
#2: Run paraprof --writecomm <heatmapName.csv> <TAU profile input>
#3: Open ParaView and do Macros->Add New Macro. Select TAU-CSV-To-HeatMap.py from its location in <tau2>/etc (This only needs to be done once.)
#4: Do File->Open in ParaView and select <heatmapName.csv>. Hit 'apply' to make sure the csv is loaded, and make sure it is selected in ParaView's pipeline browser.
#5: Run the TAU-CSV-To-HeatMap macro to display the heat map. It should be possible to select different scalar values from the ParaView UI and mouse-over nodes to see specific values by entering ParaView's 'hover points' mode.



#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'CSV Reader'
#csvSource = CSVReader(FileName=['/home/wspear/PerformanceData/BloodBridges/bloodmatrix.csv'])


csvSource = GetActiveSource()

if(('FileName' in dir(csvSource) and csvSource.FileName[0].lower().endswith(".csv")) is False):
	raise TypeError, "This macro requires a .csv file to be selected in the pipeline browser."

data = servermanager.Fetch(csvSource,0)
col0 = data.GetColumn(0)
maxrank = col0.GetValueRange()[1]

firstScalarName = data.GetColumnName(3)
firstScalarMaxValue = data.GetColumn(3).GetValueRange()[1]

zScaleFactor= maxrank*1.0/firstScalarMaxValue


csvSource.AddTabFieldDelimiter = 1

# Create a new 'SpreadSheet View'
#spreadSheetView1 = CreateView('SpreadSheetView')
#spreadSheetView1.ColumnToSort = ''
#spreadSheetView1.BlockSize = 1024L
# uncomment following to set a specific view size
# spreadSheetView1.ViewSize = [400, 400]

# get layout
#layout1 = GetLayout()

# place view in the layout
#layout1.AssignView(2, spreadSheetView1)

# show data in view
#csvSourceDisplay = Show(csvSource, spreadSheetView1)

# trace defaults for the display properties.
#csvSourceDisplay.FieldAssociation = 'Row Data'

# update the view to ensure updated data information
#spreadSheetView1.Update()

# create a new 'Table To Points'
tableToPoints1 = TableToPoints(Input=csvSource)
#tableToPoints1.XColumn = ' All Paths CALLS'
#tableToPoints1.YColumn = ' All Paths CALLS'
#tableToPoints1.ZColumn = ' All Paths CALLS'

# Properties modified on tableToPoints1
tableToPoints1.XColumn = 'x cord'
tableToPoints1.YColumn = 'y coord'
tableToPoints1.ZColumn = 'z coord'
tableToPoints1.a2DPoints = 1

# show data in view
#tableToPoints1Display = Show(tableToPoints1, spreadSheetView1)

# hide data in view
#Hide(csvSource, spreadSheetView1)

# update the view to ensure updated data information
#spreadSheetView1.Update()

# create a new 'Warp By Scalar'
warpByScalar1 = WarpByScalar(Input=tableToPoints1)
warpByScalar1.Scalars = ['POINTS', firstScalarName]

# show data in view
warpByScalar1Display = Show(warpByScalar1)#, spreadSheetView1)

# hide data in view
#Hide(tableToPoints1, spreadSheetView1)

# update the view to ensure updated data information
#spreadSheetView1.Update()

# find view
renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1058, 1361]

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(warpByScalar1)

# show data in view
warpByScalar1Display_1 = Show(warpByScalar1, renderView1)

# trace defaults for the display properties.
#warpByScalar1Display_1.Representation = 'Surface'
warpByScalar1Display_1.ColorArrayName = [None, '']
warpByScalar1Display_1.OSPRayScaleArray = firstScalarName
warpByScalar1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
warpByScalar1Display_1.SelectOrientationVectors = firstScalarName
warpByScalar1Display_1.ScaleFactor = 106.7
warpByScalar1Display_1.SelectScaleArray = firstScalarName
warpByScalar1Display_1.GlyphType = 'Box'
warpByScalar1Display_1.GlyphTableIndexArray = firstScalarName
warpByScalar1Display_1.DataAxesGrid = 'GridAxesRepresentation'
warpByScalar1Display_1.PolarAxes = 'PolarAxesRepresentation'
warpByScalar1Display_1.GaussianRadius = 53.35
warpByScalar1Display_1.SetScaleArray = ['POINTS', firstScalarName]
warpByScalar1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
warpByScalar1Display_1.OpacityArray = ['POINTS', firstScalarName]
warpByScalar1Display_1.OpacityTransferFunction = 'PiecewiseFunction'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
warpByScalar1Display_1.ScaleTransferFunction.Points = [33.0, 0.0, 0.5, 0.0, 1100.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
warpByScalar1Display_1.OpacityTransferFunction.Points = [33.0, 0.0, 0.5, 0.0, 1100.0, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera()

# set scalar coloring
ColorBy(warpByScalar1Display_1, ('POINTS', firstScalarName))

# rescale color and/or opacity maps used to include current data range
warpByScalar1Display_1.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
warpByScalar1Display_1.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for firstScalarName
firstScalarNameLUT = GetColorTransferFunction(firstScalarName)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
#spreadSheetView1.Update()

# reset view to fit data
renderView1.ResetCamera()

# reset view to fit data bounds
renderView1.ResetCamera(0.0, 15.0, 0.0, 15.0, 3.29999995232, 110.0)

# Properties modified on warpByScalar1
warpByScalar1.ScaleFactor = zScaleFactor

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
#spreadSheetView1.Update()

# reset view to fit data
renderView1.ResetCamera()

if(maxrank < 100):
# change representation type
	warpByScalar1Display_1.SetRepresentationType('3D Glyphs')
# Properties modified on warpByScalar1Display_1
	warpByScalar1Display_1.GlyphType = 'Box'
else:
	warpByScalar1Display_1.SetRepresentationType('Points')
	warpByScalar1Display_1.PointSize = 5.0

# Properties modified on renderView1.AxesGrid
renderView1.AxesGrid.XTitle = 'Sender'
renderView1.AxesGrid.YTitle = 'Receiver'
renderView1.AxesGrid.ZTitle = firstScalarName #'Scalar'
renderView1.AxesGrid.DataScale = [1.0, 1.0, zScaleFactor]
renderView1.AxesGrid.DataBoundsInflateFactor = 0.0

# Properties modified on renderView1.AxesGrid
renderView1.AxesGrid.Visibility = 1

#### saving camera placements for all active views

# current camera placement for renderView1
#renderView1.CameraPosition = [61.83291456160151, 12.611393192578767, 25.95584146136273]
#renderView1.CameraFocalPoint = [7.5000000000000115, 7.5, 5.665000006556516]
#renderView1.CameraViewUp = [-0.35937508048268185, 0.23675820256225832, 0.9026600163115427]
#renderView1.CameraParallelScale = 15.272981668413909

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).

