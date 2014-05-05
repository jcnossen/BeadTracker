<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="11008008">
	<Property Name="CCSymbols" Type="Str"></Property>
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">true</Property>
	<Property Name="NI.Project.Description" Type="Str"></Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="NI.SortType" Type="Int">3</Property>
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="Modules" Type="Folder">
			<Property Name="NI.SortType" Type="Int">3</Property>
			<Item Name="Cameras" Type="Folder">
				<Item Name="CameraInterface" Type="Folder">
					<Item Name="CameraInterface.ctl" Type="VI" URL="../Modules/CameraInterface/CameraInterface.ctl"/>
					<Item Name="CIAdjustBeadPos.ctl" Type="VI" URL="../Modules/CameraInterface/CIAdjustBeadPos.ctl"/>
					<Item Name="CICloseType.ctl" Type="VI" URL="../Modules/CameraInterface/CICloseType.ctl"/>
					<Item Name="CIGetSetGenericConfig.ctl" Type="VI" URL="../Modules/CameraInterface/CIGetSetGenericConfig.ctl"/>
					<Item Name="CIGrabType.ctl" Type="VI" URL="../Modules/CameraInterface/CIGrabType.ctl"/>
					<Item Name="CISaveLoadSettings.ctl" Type="VI" URL="../Modules/CameraInterface/CISaveLoadSettings.ctl"/>
					<Item Name="CISettingsDlg.ctl" Type="VI" URL="../Modules/CameraInterface/CISettingsDlg.ctl"/>
				</Item>
				<Item Name="GenericIMAQCamera" Type="Folder"/>
			</Item>
			<Item Name="PIMotorController" Type="Folder"/>
		</Item>
		<Item Name="QTrkLVBinding" Type="Folder" URL="../qtrk/QTrkLVBinding">
			<Property Name="NI.DISK" Type="Bool">true</Property>
			<Property Name="NI.SortType" Type="Int">3</Property>
		</Item>
		<Item Name="Main" Type="Folder" URL="../Main">
			<Property Name="NI.DISK" Type="Bool">true</Property>
		</Item>
		<Item Name="BeadTrackerMain.vi" Type="VI" URL="../BeadTrackerMain.vi"/>
		<Item Name="SetupConfiguration.vi" Type="VI" URL="../SetupConfiguration.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="BuildHelpPath.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/BuildHelpPath.vi"/>
				<Item Name="Check if File or Folder Exists.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/libraryn.llb/Check if File or Folder Exists.vi"/>
				<Item Name="Check Special Tags.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Check Special Tags.vi"/>
				<Item Name="Clear Errors.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Clear Errors.vi"/>
				<Item Name="Close File+.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Close File+.vi"/>
				<Item Name="compatReadText.vi" Type="VI" URL="/&lt;vilib&gt;/_oldvers/_oldvers.llb/compatReadText.vi"/>
				<Item Name="Convert property node font to graphics font.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Convert property node font to graphics font.vi"/>
				<Item Name="Details Display Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Details Display Dialog.vi"/>
				<Item Name="DialogType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/DialogType.ctl"/>
				<Item Name="DialogTypeEnum.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/DialogTypeEnum.ctl"/>
				<Item Name="Error Cluster From Error Code.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Cluster From Error Code.vi"/>
				<Item Name="Error Code Database.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Code Database.vi"/>
				<Item Name="ErrWarn.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/ErrWarn.ctl"/>
				<Item Name="eventvkey.ctl" Type="VI" URL="/&lt;vilib&gt;/event_ctls.llb/eventvkey.ctl"/>
				<Item Name="ex_CorrectErrorChain.vi" Type="VI" URL="/&lt;vilib&gt;/express/express shared/ex_CorrectErrorChain.vi"/>
				<Item Name="Find First Error.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Find First Error.vi"/>
				<Item Name="Find Tag.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Find Tag.vi"/>
				<Item Name="Format Message String.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Format Message String.vi"/>
				<Item Name="General Error Handler CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/General Error Handler CORE.vi"/>
				<Item Name="General Error Handler.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/General Error Handler.vi"/>
				<Item Name="Get String Text Bounds.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Get String Text Bounds.vi"/>
				<Item Name="Get Text Rect.vi" Type="VI" URL="/&lt;vilib&gt;/picture/picture.llb/Get Text Rect.vi"/>
				<Item Name="GetHelpDir.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/GetHelpDir.vi"/>
				<Item Name="GetRTHostConnectedProp.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/GetRTHostConnectedProp.vi"/>
				<Item Name="Image Type" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Image Type"/>
				<Item Name="IMAQ ArrayToImage" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ArrayToImage"/>
				<Item Name="IMAQ Close.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Close.vi"/>
				<Item Name="IMAQ Configure Buffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Configure Buffer.vi"/>
				<Item Name="IMAQ Configure List.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Configure List.vi"/>
				<Item Name="IMAQ Copy" Type="VI" URL="/&lt;vilib&gt;/vision/Management.llb/IMAQ Copy"/>
				<Item Name="IMAQ Create" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Create"/>
				<Item Name="IMAQ Dispose" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Dispose"/>
				<Item Name="IMAQ Extract Buffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Extract Buffer.vi"/>
				<Item Name="IMAQ GetImagePixelPtr" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ GetImagePixelPtr"/>
				<Item Name="IMAQ GetImageSize" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ GetImageSize"/>
				<Item Name="IMAQ Image.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/IMAQ Image.ctl"/>
				<Item Name="IMAQ ImageToArray" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ImageToArray"/>
				<Item Name="IMAQ Init.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Init.vi"/>
				<Item Name="IMAQ Serial Read.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Serial Read.vi"/>
				<Item Name="IMAQ Serial Write.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Serial Write.vi"/>
				<Item Name="IMAQ Start.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Start.vi"/>
				<Item Name="IMAQ Stop.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Stop.vi"/>
				<Item Name="IMAQ Write BMP File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write BMP File 2"/>
				<Item Name="IMAQ Write File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write File 2"/>
				<Item Name="IMAQ Write Image And Vision Info File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write Image And Vision Info File 2"/>
				<Item Name="IMAQ Write JPEG File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write JPEG File 2"/>
				<Item Name="IMAQ Write JPEG2000 File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write JPEG2000 File 2"/>
				<Item Name="IMAQ Write PNG File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write PNG File 2"/>
				<Item Name="IMAQ Write TIFF File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write TIFF File 2"/>
				<Item Name="Longest Line Length in Pixels.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Longest Line Length in Pixels.vi"/>
				<Item Name="LVBoundsTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVBoundsTypeDef.ctl"/>
				<Item Name="LVPointDoubleTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVPointDoubleTypeDef.ctl"/>
				<Item Name="NI_AALBase.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALBase.lvlib"/>
				<Item Name="NI_AALPro.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALPro.lvlib"/>
				<Item Name="NI_FileType.lvlib" Type="Library" URL="/&lt;vilib&gt;/Utility/lvfile.llb/NI_FileType.lvlib"/>
				<Item Name="NI_Matrix.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/Matrix/NI_Matrix.lvlib"/>
				<Item Name="NI_PackedLibraryUtility.lvlib" Type="Library" URL="/&lt;vilib&gt;/Utility/LVLibp/NI_PackedLibraryUtility.lvlib"/>
				<Item Name="NI_Vision_Development_Module.lvlib" Type="Library" URL="/&lt;vilib&gt;/vision/NI_Vision_Development_Module.lvlib"/>
				<Item Name="Not Found Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Not Found Dialog.vi"/>
				<Item Name="Open File+.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Open File+.vi"/>
				<Item Name="Read File+ (string).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read File+ (string).vi"/>
				<Item Name="Read From Spreadsheet File (DBL).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read From Spreadsheet File (DBL).vi"/>
				<Item Name="Read From Spreadsheet File (I64).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read From Spreadsheet File (I64).vi"/>
				<Item Name="Read From Spreadsheet File (string).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read From Spreadsheet File (string).vi"/>
				<Item Name="Read From Spreadsheet File.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read From Spreadsheet File.vi"/>
				<Item Name="Read Lines From File.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read Lines From File.vi"/>
				<Item Name="Search and Replace Pattern.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Search and Replace Pattern.vi"/>
				<Item Name="Set Bold Text.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set Bold Text.vi"/>
				<Item Name="Set String Value.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set String Value.vi"/>
				<Item Name="subFile Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/express/express input/FileDialogBlock.llb/subFile Dialog.vi"/>
				<Item Name="TagReturnType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/TagReturnType.ctl"/>
				<Item Name="Three Button Dialog CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog CORE.vi"/>
				<Item Name="Three Button Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog.vi"/>
				<Item Name="Trim Whitespace.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Trim Whitespace.vi"/>
				<Item Name="VISA Configure Serial Port" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port"/>
				<Item Name="VISA Configure Serial Port (Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Instr).vi"/>
				<Item Name="VISA Configure Serial Port (Serial Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Serial Instr).vi"/>
				<Item Name="whitespace.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/whitespace.ctl"/>
				<Item Name="Write Spreadsheet String.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write Spreadsheet String.vi"/>
				<Item Name="Write To Spreadsheet File (DBL).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File (DBL).vi"/>
				<Item Name="Write To Spreadsheet File (I64).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File (I64).vi"/>
				<Item Name="Write To Spreadsheet File (string).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File (string).vi"/>
				<Item Name="Write To Spreadsheet File.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File.vi"/>
				<Item Name="SessionLookUp.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/SessionLookUp.vi"/>
				<Item Name="imgUpdateErrorCluster.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgUpdateErrorCluster.vi"/>
				<Item Name="imgInterfaceOpen.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgInterfaceOpen.vi"/>
				<Item Name="imgSessionOpen.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionOpen.vi"/>
				<Item Name="IMAQRegisterSession.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/IMAQRegisterSession.vi"/>
				<Item Name="imgClose.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgClose.vi"/>
				<Item Name="IMAQUnregisterSession.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/IMAQUnregisterSession.vi"/>
				<Item Name="imgSessionStopAcquisition.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionStopAcquisition.vi"/>
				<Item Name="imgEnsureEqualBorders.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgEnsureEqualBorders.vi"/>
				<Item Name="imgGetBufList.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgGetBufList.vi"/>
				<Item Name="imgMemLock.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgMemLock.vi"/>
				<Item Name="imgSessionConfigure.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionConfigure.vi"/>
				<Item Name="imgSessionAcquire.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionAcquire.vi"/>
				<Item Name="imgCreateBufList.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgCreateBufList.vi"/>
				<Item Name="imgSetRoi.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSetRoi.vi"/>
				<Item Name="imgSessionAttribute.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionAttribute.vi"/>
				<Item Name="imgBufferElement.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgBufferElement.vi"/>
				<Item Name="imgDisposeBufList.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgDisposeBufList.vi"/>
				<Item Name="imgIsNewStyleInterface.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgIsNewStyleInterface.vi"/>
				<Item Name="IMAQ Extract Buffer Old Style.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/IMAQ Extract Buffer Old Style.vi"/>
				<Item Name="imgSessionReleaseBuffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionReleaseBuffer.vi"/>
				<Item Name="imgSessionExamineBuffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionExamineBuffer.vi"/>
				<Item Name="imgWaitForIMAQOccurrence.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgWaitForIMAQOccurrence.vi"/>
				<Item Name="IMAQ Attribute.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Attribute.vi"/>
				<Item Name="Application Directory.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Application Directory.vi"/>
				<Item Name="IMAQ ReadFile" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ ReadFile"/>
				<Item Name="IMAQ Clear Overlay" Type="VI" URL="/&lt;vilib&gt;/vision/Overlay.llb/IMAQ Clear Overlay"/>
				<Item Name="IMAQ Overlay Rectangle" Type="VI" URL="/&lt;vilib&gt;/vision/Overlay.llb/IMAQ Overlay Rectangle"/>
				<Item Name="Color to RGB.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/colorconv.llb/Color to RGB.vi"/>
				<Item Name="LVPoint32TypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVPoint32TypeDef.ctl"/>
				<Item Name="IMAQ Overlay Text" Type="VI" URL="/&lt;vilib&gt;/vision/Overlay.llb/IMAQ Overlay Text"/>
				<Item Name="IMAQ Merge Overlay" Type="VI" URL="/&lt;vilib&gt;/vision/Overlay.llb/IMAQ Merge Overlay"/>
				<Item Name="Obtain Semaphore Reference.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/semaphor.llb/Obtain Semaphore Reference.vi"/>
				<Item Name="Semaphore RefNum" Type="VI" URL="/&lt;vilib&gt;/Utility/semaphor.llb/Semaphore RefNum"/>
				<Item Name="Semaphore Refnum Core.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/semaphor.llb/Semaphore Refnum Core.ctl"/>
				<Item Name="AddNamedSemaphorePrefix.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/semaphor.llb/AddNamedSemaphorePrefix.vi"/>
				<Item Name="GetNamedSemaphorePrefix.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/semaphor.llb/GetNamedSemaphorePrefix.vi"/>
				<Item Name="Validate Semaphore Size.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/semaphor.llb/Validate Semaphore Size.vi"/>
				<Item Name="Acquire Semaphore.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/semaphor.llb/Acquire Semaphore.vi"/>
				<Item Name="Release Semaphore.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/semaphor.llb/Release Semaphore.vi"/>
				<Item Name="Not A Semaphore.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/semaphor.llb/Not A Semaphore.vi"/>
				<Item Name="System Exec.vi" Type="VI" URL="/&lt;vilib&gt;/Platform/system.llb/System Exec.vi"/>
				<Item Name="IMAQ SetImageSize" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ SetImageSize"/>
			</Item>
			<Item Name="kernel32.dll" Type="Document" URL="kernel32.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivision.dll" Type="Document" URL="nivision.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="imaq.dll" Type="Document" URL="imaq.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="FinderDlg.vi" Type="VI" URL="../AutoBeadFinderQTrk.llb/FinderDlg.vi"/>
			<Item Name="DrawUI_Image.vi" Type="VI" URL="../AutoBeadFinderQTrk.llb/DrawUI_Image.vi"/>
			<Item Name="RemoveClosestBead.vi" Type="VI" URL="../AutoBeadFinderQTrk.llb/RemoveClosestBead.vi"/>
			<Item Name="ComputeLocalCOM.vi" Type="VI" URL="../AutoBeadFinderQTrk.llb/ComputeLocalCOM.vi"/>
			<Item Name="Compute2DArrayCOM.vi" Type="VI" URL="../AutoBeadFinderQTrk.llb/Compute2DArrayCOM.vi"/>
			<Item Name="GCS_Interface.ctl" Type="VI" URL="../Modules/PIMotorController/GCS_Interface.ctl"/>
			<Item Name="PI_Axis_list.ctl" Type="VI" URL="../Modules/PIMotorController/PI_Axis_list.ctl"/>
			<Item Name="PI_Axis.ctl" Type="VI" URL="../Modules/PIMotorController/PI_Axis.ctl"/>
			<Item Name="PIMotorController.vi" Type="VI" URL="../Modules/PIMotorController/PIMotorController.vi"/>
			<Item Name="AxisEnumToAxisInfo.vi" Type="VI" URL="../Modules/PIMotorController/AxisEnumToAxisInfo.vi"/>
			<Item Name="PIAxisInfoGlobal.vi" Type="VI" URL="../Modules/PIMotorController/PIAxisInfoGlobal.vi"/>
			<Item Name="MeasureCurrentPos.vi" Type="VI" URL="../Modules/PIMotorController/MeasureCurrentPos.vi"/>
			<Item Name="GetSingleAxisPos.vi" Type="VI" URL="../Modules/PIMotorController/GetSingleAxisPos.vi"/>
			<Item Name="Cmd_SetPosition.vi" Type="VI" URL="../Modules/PIMotorController/Cmd_SetPosition.vi"/>
			<Item Name="MoveSingleAxis.vi" Type="VI" URL="../Modules/PIMotorController/MoveSingleAxis.vi"/>
			<Item Name="Cmd_UpdatePositions.vi" Type="VI" URL="../Modules/PIMotorController/Cmd_UpdatePositions.vi"/>
			<Item Name="Cmd_MoveToLimit.vi" Type="VI" URL="../Modules/PIMotorController/Cmd_MoveToLimit.vi"/>
			<Item Name="ROIlistautofill.vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/ROIlistautofill.vi"/>
			<Item Name="xy2roi.vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/xy2roi.vi"/>
			<Item Name="draw rectangles.vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/draw rectangles.vi"/>
			<Item Name="CleanROIs (SubVI).vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/CleanROIs (SubVI).vi"/>
			<Item Name="RECenterROI (SubVI).vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/RECenterROI (SubVI).vi"/>
			<Item Name="XY_GetCenterOfMass.vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/XY_GetCenterOfMass.vi"/>
			<Item Name="ROIAutoSearch.vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/ROIAutoSearch.vi"/>
			<Item Name="MakeBigTemplate (SubVI).vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/MakeBigTemplate (SubVI).vi"/>
			<Item Name="MinusMean2D (SubVI).vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/MinusMean2D (SubVI).vi"/>
			<Item Name="Xcorimages (SubVI).vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/Xcorimages (SubVI).vi"/>
			<Item Name="Swapit2D (SubVI).vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/Swapit2D (SubVI).vi"/>
			<Item Name="CleanIT (SubVI).vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/CleanIT (SubVI).vi"/>
			<Item Name="SortOnKey (SubVI).vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/SortOnKey (SubVI).vi"/>
			<Item Name="Select Bests (SubVI).vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/Select Bests (SubVI).vi"/>
			<Item Name="ROICenter2LTRB.vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/ROICenter2LTRB.vi"/>
			<Item Name="RemovenearestROI.vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/RemovenearestROI.vi"/>
			<Item Name="roi2xy.vi" Type="VI" URL="../OfflineTracker/AutoBeadFinder.llb/roi2xy.vi"/>
			<Item Name="MoveOutlierROIs.vi" Type="VI" URL="../AutoBeadFinderQTrk.llb/MoveOutlierROIs.vi"/>
			<Item Name="lvanlys.dll" Type="Document" URL="/C/Program Files/National Instruments/LabVIEW 2011/resource/lvanlys.dll"/>
			<Item Name="D012L_Motors_E753.vi" Type="VI" URL="../Setups/D012L/D012L_Motors_E753.vi"/>
			<Item Name="D012L_Motors_init_all.vi" Type="VI" URL="../Setups/D012L/MotorsInit-SerialE753.llb/D012L_Motors_init_all.vi"/>
			<Item Name="Motors_Mercury_init_D012_MD.vi" Type="VI" URL="../Setups/D012L/MotorsInit.llb/Motors_Mercury_init_D012_MD.vi"/>
			<Item Name="Mercury_GCS_Configuration_Setup.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Mercury_GCS_Configuration_Setup.vi"/>
			<Item Name="Available DLLs.ctl" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Available DLLs.ctl"/>
			<Item Name="Available interfaces.ctl" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Available interfaces.ctl"/>
			<Item Name="Available DLL interfaces.ctl" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Available DLL interfaces.ctl"/>
			<Item Name="Controller names.ctl" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/Controller names.ctl"/>
			<Item Name="#24.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Special command.llb/#24.vi"/>
			<Item Name="PI Send String.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/PI Send String.vi"/>
			<Item Name="Termination character.ctl" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Termination character.ctl"/>
			<Item Name="Global1.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Global1.vi"/>
			<Item Name="GCSTranslator DLL Functions.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/GCSTranslator DLL Functions.vi"/>
			<Item Name="GCSTranslator.dll" Type="Document" URL="../Setups/D012L/PI Drivers/Low Level/GCSTranslator.dll"/>
			<Item Name="Global DaisyChain.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Global DaisyChain.vi"/>
			<Item Name="Get lines from string.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Get lines from string.vi"/>
			<Item Name="Cut out additional spaces.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Cut out additional spaces.vi"/>
			<Item Name="Analog Functions.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Analog control.llb/Analog Functions.vi"/>
			<Item Name="Available Analog Commands.ctl" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Analog control.llb/Available Analog Commands.ctl"/>
			<Item Name="String with ASCII code conversion.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/String with ASCII code conversion.vi"/>
			<Item Name="Initialize Global1.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Initialize Global1.vi"/>
			<Item Name="Initialize Global DaisyChain.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Initialize Global DaisyChain.vi"/>
			<Item Name="Initialize Global2.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/Initialize Global2.vi"/>
			<Item Name="Global2 (Array).vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/Global2 (Array).vi"/>
			<Item Name="Select USB device.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Select USB device.vi"/>
			<Item Name="GCSTranslateError.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/GCSTranslateError.vi"/>
			<Item Name="Select DaisyChain device.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Select DaisyChain device.vi"/>
			<Item Name="Is DaisyChain open.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Is DaisyChain open.vi"/>
			<Item Name="PI Open Interface of one system.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/PI Open Interface of one system.vi"/>
			<Item Name="Close connection if open.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Close connection if open.vi"/>
			<Item Name="Analog FGlobal.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Analog control.llb/Analog FGlobal.vi"/>
			<Item Name="PI Receive String.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/PI Receive String.vi"/>
			<Item Name="Analog Receive String.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Analog control.llb/Analog Receive String.vi"/>
			<Item Name="PI VISA Receive Characters.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/PI VISA Receive Characters.vi"/>
			<Item Name="*IDN?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/*IDN?.vi"/>
			<Item Name="Define connected systems (Array).vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/Define connected systems (Array).vi"/>
			<Item Name="Define connected axes.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/Define connected axes.vi"/>
			<Item Name="SAI?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/SAI?.vi"/>
			<Item Name="Substract axes array subset from axes array.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Substract axes array subset from axes array.vi"/>
			<Item Name="Commanded axes connected?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Commanded axes connected?.vi"/>
			<Item Name="Get all axes.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Get all axes.vi"/>
			<Item Name="Get arrays without blanks.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Get arrays without blanks.vi"/>
			<Item Name="CST?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Special command.llb/CST?.vi"/>
			<Item Name="Build query command substring.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Build query command substring.vi"/>
			<Item Name="Return space.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Return space.vi"/>
			<Item Name="Assign values from string to axes.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Assign values from string to axes.vi"/>
			<Item Name="SVO?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/SVO?.vi"/>
			<Item Name="Assign booleans from string to axes.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Assign booleans from string to axes.vi"/>
			<Item Name="SVO.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/SVO.vi"/>
			<Item Name="Build command substring.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Build command substring.vi"/>
			<Item Name="Set RON and return RON status.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Set RON and return RON status.vi"/>
			<Item Name="RON?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/RON?.vi"/>
			<Item Name="Define axes to command from boolean array.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Define axes to command from boolean array.vi"/>
			<Item Name="RON.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/RON.vi"/>
			<Item Name="FRF?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/FRF?.vi"/>
			<Item Name="ERR?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/ERR?.vi"/>
			<Item Name="POS?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/POS?.vi"/>
			<Item Name="Assign NaN for chosen axes.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Assign NaN for chosen axes.vi"/>
			<Item Name="TRS?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/TRS?.vi"/>
			<Item Name="FRF.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/FRF.vi"/>
			<Item Name="Wait for controller ready.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Wait for controller ready.vi"/>
			<Item Name="#7.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Special command.llb/#7.vi"/>
			<Item Name="LIM?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/LIM?.vi"/>
			<Item Name="FPL.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/FPL.vi"/>
			<Item Name="FNL.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/FNL.vi"/>
			<Item Name="TMX?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/TMX?.vi"/>
			<Item Name="TMN?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/TMN?.vi"/>
			<Item Name="Select values for chosen axes.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Select values for chosen axes.vi"/>
			<Item Name="MOV.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/MOV.vi"/>
			<Item Name="General wait for movement to stop.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/General wait for movement to stop.vi"/>
			<Item Name="Global Analog.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Analog control.llb/Global Analog.vi"/>
			<Item Name="Wait for axes to stop.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Wait for axes to stop.vi"/>
			<Item Name="#5.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Special command.llb/#5.vi"/>
			<Item Name="STA?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Special command.llb/STA?.vi"/>
			<Item Name="ONT?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/ONT?.vi"/>
			<Item Name="Wait for hexapod system axes to stop.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Old commands.llb/Wait for hexapod system axes to stop.vi"/>
			<Item Name="#5_old.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Old commands.llb/#5_old.vi"/>
			<Item Name="Motors_Init_E753.vi" Type="VI" URL="../Setups/D012L/MotorsInit-SerialE753.llb/Motors_Init_E753.vi"/>
			<Item Name="E712_Configuration_Setup.vi" Type="VI" URL="../Setups/D012L/MotorsInit.llb/E712_Configuration_Setup.vi"/>
			<Item Name="Select host address.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Select host address.vi"/>
			<Item Name="Find host address.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Find host address.vi"/>
			<Item Name="Get subnet.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Communication.llb/Get subnet.vi"/>
			<Item Name="#9.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/WaveGenerator.llb/#9.vi"/>
			<Item Name="TWG?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/WaveGenerator.llb/TWG?.vi"/>
			<Item Name="WGO.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/WaveGenerator.llb/WGO.vi"/>
			<Item Name="HLP?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/General command.llb/HLP?.vi"/>
			<Item Name="Combine axes arrays.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Combine axes arrays.vi"/>
			<Item Name="ATZ.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/ATZ.vi"/>
			<Item Name="Longlasting one-axis command.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Longlasting one-axis command.vi"/>
			<Item Name="Wait for answer of longlasting command.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Support.llb/Wait for answer of longlasting command.vi"/>
			<Item Name="ATZ?.vi" Type="VI" URL="../Setups/D012L/PI Drivers/Low Level/Limits.llb/ATZ?.vi"/>
			<Item Name="LoadGCSLowLevelVIs.vi" Type="VI" URL="../Modules/PIMotorController/LoadGCSLowLevelVIs.vi"/>
			<Item Name="ConnectToCamera.vi" Type="VI" URL="../Modules/MikrotronCamera/ConnectToCamera.vi"/>
			<Item Name="CIInit.vi" Type="VI" URL="../Modules/MikrotronCamera/CIInit.vi"/>
			<Item Name="CameraConfig.ctl" Type="VI" URL="../Modules/MikrotronCamera/CameraConfig.ctl"/>
			<Item Name="mkCameraData.ctl" Type="VI" URL="../Modules/MikrotronCamera/mkCameraData.ctl"/>
			<Item Name="CIGetSetConfig.vi" Type="VI" URL="../Modules/MikrotronCamera/CIGetSetConfig.vi"/>
			<Item Name="CIClose.vi" Type="VI" URL="../Modules/MikrotronCamera/CIClose.vi"/>
			<Item Name="CISaveLoadSettings.vi" Type="VI" URL="../Modules/MikrotronCamera/CISaveLoadSettings.vi"/>
			<Item Name="CIGrab.vi" Type="VI" URL="../Modules/MikrotronCamera/CIGrab.vi"/>
			<Item Name="mkSetROIs.vi" Type="VI" URL="../Modules/MikrotronCamera/mkSetROIs.vi"/>
			<Item Name="mkSendSerialCmd.vi" Type="VI" URL="../Modules/MikrotronCamera/mkSendSerialCmd.vi"/>
			<Item Name="mkGetSetInFrameCounter.vi" Type="VI" URL="../Modules/MikrotronCamera/mkGetSetInFrameCounter.vi"/>
			<Item Name="mkGetROIs.vi" Type="VI" URL="../Modules/MikrotronCamera/mkGetROIs.vi"/>
			<Item Name="mkROIsToString.vi" Type="VI" URL="../Modules/MikrotronCamera/mkROIsToString.vi"/>
			<Item Name="mkGetSetFramerate.vi" Type="VI" URL="../Modules/MikrotronCamera/mkGetSetFramerate.vi"/>
			<Item Name="mkSetExposureGainOffset.vi" Type="VI" URL="../Modules/MikrotronCamera/mkSetExposureGainOffset.vi"/>
			<Item Name="mkConfigureBufferList.vi" Type="VI" URL="../Modules/MikrotronCamera/mkConfigureBufferList.vi"/>
			<Item Name="GrabToTrackerAndQueue(LostFramesSupport).vi" Type="VI" URL="../Modules/MikrotronCamera/GrabToTrackerAndQueue(LostFramesSupport).vi"/>
			<Item Name="GrabToTrackerFastLoop.vi" Type="VI" URL="../Modules/MikrotronCamera/GrabToTrackerFastLoop.vi"/>
			<Item Name="CIAdjustBeadPos.vi" Type="VI" URL="../Modules/MikrotronCamera/CIAdjustBeadPos.vi"/>
			<Item Name="mkAdjustROIPositions.vi" Type="VI" URL="../Modules/MikrotronCamera/mkAdjustROIPositions.vi"/>
			<Item Name="CISettingsEditor.vi" Type="VI" URL="../Modules/MikrotronCamera/CISettingsEditor.vi"/>
			<Item Name="mkShowSettingsDialog.vi" Type="VI" URL="../Modules/MikrotronCamera/mkShowSettingsDialog.vi"/>
			<Item Name="mkConfigure.vi" Type="VI" URL="../Modules/MikrotronCamera/mkConfigure.vi"/>
			<Item Name="mkSetMode.vi" Type="VI" URL="../Modules/MikrotronCamera/mkSetMode.vi"/>
			<Item Name="mkResetCamera.vi" Type="VI" URL="../Modules/MikrotronCamera/mkResetCamera.vi"/>
			<Item Name="FastCMOSTestTrackerModule.vi" Type="VI" URL="../Modules/FastCMOS.llb/FastCMOSTestTrackerModule.vi"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
