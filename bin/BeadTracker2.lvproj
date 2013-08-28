<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="11008008">
	<Property Name="CCSymbols" Type="Str"></Property>
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">true</Property>
	<Property Name="NI.Project.Description" Type="Str"></Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="AutoBeadFinder" Type="Folder">
			<Item Name="CleanIT (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/CleanIT (SubVI).vi"/>
			<Item Name="CleanROIs (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/CleanROIs (SubVI).vi"/>
			<Item Name="draw rectangles.vi" Type="VI" URL="../AutoBeadFinder.llb/draw rectangles.vi"/>
			<Item Name="MakeBigTemplate (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/MakeBigTemplate (SubVI).vi"/>
			<Item Name="MinusMean2D (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/MinusMean2D (SubVI).vi"/>
			<Item Name="RECenterROI (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/RECenterROI (SubVI).vi"/>
			<Item Name="RemovenearestROI.vi" Type="VI" URL="../AutoBeadFinder.llb/RemovenearestROI.vi"/>
			<Item Name="roi2xy.vi" Type="VI" URL="../AutoBeadFinder.llb/roi2xy.vi"/>
			<Item Name="ROIAutoSearch.vi" Type="VI" URL="../AutoBeadFinder.llb/ROIAutoSearch.vi"/>
			<Item Name="ROICenter2LTRB.vi" Type="VI" URL="../AutoBeadFinder.llb/ROICenter2LTRB.vi"/>
			<Item Name="ROIlistautofill.vi" Type="VI" URL="../AutoBeadFinder.llb/ROIlistautofill.vi"/>
			<Item Name="Select Bests (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/Select Bests (SubVI).vi"/>
			<Item Name="SortOnKey (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/SortOnKey (SubVI).vi"/>
			<Item Name="Swapit2D (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/Swapit2D (SubVI).vi"/>
			<Item Name="Xcorimages (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/Xcorimages (SubVI).vi"/>
			<Item Name="xy2roi.vi" Type="VI" URL="../AutoBeadFinder.llb/xy2roi.vi"/>
			<Item Name="XY_GetCenterOfMass.vi" Type="VI" URL="../AutoBeadFinder.llb/XY_GetCenterOfMass.vi"/>
		</Item>
		<Item Name="BeadTracker2.llb" Type="Folder">
			<Item Name="ExperimentProgram" Type="Folder">
				<Item Name="ExperimentProgramType.ctl" Type="VI" URL="../BeadTracker2.llb/ExperimentProgramType.ctl"/>
				<Item Name="ExperimentProgramUI.vi" Type="VI" URL="../BeadTracker2.llb/ExperimentProgramUI.vi"/>
				<Item Name="ExProg_InitScriptState.vi" Type="VI" URL="../BeadTracker2.llb/ExProg_InitScriptState.vi"/>
				<Item Name="ExProg_ParseFromScript.vi" Type="VI" URL="../BeadTracker2.llb/ExProg_ParseFromScript.vi"/>
				<Item Name="ExProg_Simulate.vi" Type="VI" URL="../BeadTracker2.llb/ExProg_Simulate.vi"/>
				<Item Name="ExProg_Tick.vi" Type="VI" URL="../BeadTracker2.llb/ExProg_Tick.vi"/>
				<Item Name="ExProgCommandEnumType.ctl" Type="VI" URL="../BeadTracker2.llb/ExProgCommandEnumType.ctl"/>
				<Item Name="ExProgCommandType.ctl" Type="VI" URL="../BeadTracker2.llb/ExProgCommandType.ctl"/>
				<Item Name="ExProgState.ctl" Type="VI" URL="../BeadTracker2.llb/ExProgState.ctl"/>
				<Item Name="IsMotorPosWithinRange.vi" Type="VI" URL="../BeadTracker2.llb/IsMotorPosWithinRange.vi"/>
				<Item Name="MotorAxisEnum.ctl" Type="VI" URL="../BeadTracker2.llb/MotorAxisEnum.ctl"/>
				<Item Name="RemoveCommentsFromStringArray.vi" Type="VI" URL="../BeadTracker2.llb/RemoveCommentsFromStringArray.vi"/>
				<Item Name="SplitStringIntoArray.vi" Type="VI" URL="../BeadTracker2.llb/SplitStringIntoArray.vi"/>
			</Item>
			<Item Name="Tracking" Type="Folder">
				<Item Name="BuildZLUT.vi" Type="VI" URL="../BeadTracker2.llb/BuildZLUT.vi"/>
				<Item Name="ConvertBeadPosCornerCenter.vi" Type="VI" URL="../BeadTracker2.llb/ConvertBeadPosCornerCenter.vi"/>
				<Item Name="CreateQTrkInstance.vi" Type="VI" URL="../BeadTracker2.llb/CreateQTrkInstance.vi"/>
				<Item Name="DiscardBead.vi" Type="VI" URL="../BeadTracker2.llb/DiscardBead.vi"/>
				<Item Name="EditQTrkSettingsDialog.vi" Type="VI" URL="../BeadTracker2.llb/EditQTrkSettingsDialog.vi"/>
				<Item Name="GetResultScaleAndOffset.vi" Type="VI" URL="../BeadTracker2.llb/GetResultScaleAndOffset.vi"/>
				<Item Name="SaveOrLoadBeadlist.vi" Type="VI" URL="../BeadTracker2.llb/SaveOrLoadBeadlist.vi"/>
				<Item Name="TrackerMain.vi" Type="VI" URL="../BeadTracker2.llb/TrackerMain.vi"/>
			</Item>
			<Item Name="Typedefs" Type="Folder">
				<Item Name="Cmd_CameraIn.ctl" Type="VI" URL="../BeadTracker2.llb/Cmd_CameraIn.ctl"/>
				<Item Name="Cmd_MotorIn.ctl" Type="VI" URL="../BeadTracker2.llb/Cmd_MotorIn.ctl"/>
				<Item Name="CmdData_GrabParams.ctl" Type="VI" URL="../BeadTracker2.llb/CmdData_GrabParams.ctl"/>
				<Item Name="CmdData_NewFrame.ctl" Type="VI" URL="../BeadTracker2.llb/CmdData_NewFrame.ctl"/>
				<Item Name="CmdData_SetMotorPos.ctl" Type="VI" URL="../BeadTracker2.llb/CmdData_SetMotorPos.ctl"/>
				<Item Name="CmdEnum_CameraFrameType.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_CameraFrameType.ctl"/>
				<Item Name="CmdEnum_CameraIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_CameraIn.ctl"/>
				<Item Name="CmdEnum_MotorIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_MotorIn.ctl"/>
				<Item Name="CmdEnum_UserInterfaceCmd.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_UserInterfaceCmd.ctl"/>
				<Item Name="MeasureConfigOutputMode.ctl" Type="VI" URL="../BeadTracker2.llb/MeasureConfigOutputMode.ctl"/>
				<Item Name="MeasureConfigType.ctl" Type="VI" URL="../BeadTracker2.llb/MeasureConfigType.ctl"/>
				<Item Name="MotorConfigType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorConfigType.ctl"/>
				<Item Name="MotorPosFlagsType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorPosFlagsType.ctl"/>
				<Item Name="MotorPosType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorPosType.ctl"/>
				<Item Name="MotorStateType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorStateType.ctl"/>
				<Item Name="QueueListType.ctl" Type="VI" URL="../BeadTracker2.llb/QueueListType.ctl"/>
				<Item Name="XYf.ctl" Type="VI" URL="../BeadTracker2.llb/XYf.ctl"/>
			</Item>
			<Item Name="ApplyOffsetGainToDisplayImage.vi" Type="VI" URL="../BeadTracker2.llb/ApplyOffsetGainToDisplayImage.vi"/>
			<Item Name="AskForCreateNewDir.vi" Type="VI" URL="../BeadTracker2.llb/AskForCreateNewDir.vi"/>
			<Item Name="ClearResults.vi" Type="VI" URL="../BeadTracker2.llb/ClearResults.vi"/>
			<Item Name="ClusterToValueList.vi" Type="VI" URL="../BeadTracker2.llb/ClusterToValueList.vi"/>
			<Item Name="ComputeFisherMatrixFromLUT.vi" Type="VI" URL="../BeadTracker2.llb/ComputeFisherMatrixFromLUT.vi"/>
			<Item Name="CreateNoiseCorrectionImage.vi" Type="VI" URL="../BeadTracker2.llb/CreateNoiseCorrectionImage.vi"/>
			<Item Name="DeleteFileIfExisting.vi" Type="VI" URL="../BeadTracker2.llb/DeleteFileIfExisting.vi"/>
			<Item Name="ExtractBeadROI.vi" Type="VI" URL="../BeadTracker2.llb/ExtractBeadROI.vi"/>
			<Item Name="GenerateNewFilename.vi" Type="VI" URL="../BeadTracker2.llb/GenerateNewFilename.vi"/>
			<Item Name="GenerateTracePlotData.vi" Type="VI" URL="../BeadTracker2.llb/GenerateTracePlotData.vi"/>
			<Item Name="GetExperimentPaths.vi" Type="VI" URL="../BeadTracker2.llb/GetExperimentPaths.vi"/>
			<Item Name="GetQueues.vi" Type="VI" URL="../BeadTracker2.llb/GetQueues.vi"/>
			<Item Name="GlobalVariables.vi" Type="VI" URL="../BeadTracker2.llb/GlobalVariables.vi"/>
			<Item Name="GrabSingleImage.vi" Type="VI" URL="../BeadTracker2.llb/GrabSingleImage.vi"/>
			<Item Name="LimitedMotorMoveCmd.vi" Type="VI" URL="../BeadTracker2.llb/LimitedMotorMoveCmd.vi"/>
			<Item Name="LimitMotorPos.vi" Type="VI" URL="../BeadTracker2.llb/LimitMotorPos.vi"/>
			<Item Name="LoadGainCorrectionImages.vi" Type="VI" URL="../BeadTracker2.llb/LoadGainCorrectionImages.vi"/>
			<Item Name="LogMsg.vi" Type="VI" URL="../BeadTracker2.llb/LogMsg.vi"/>
			<Item Name="MotorUI.vi" Type="VI" URL="../BeadTracker2.llb/MotorUI.vi"/>
			<Item Name="RestoreMainImageView.vi" Type="VI" URL="../BeadTracker2.llb/RestoreMainImageView.vi"/>
			<Item Name="RunSaveImagesExperiment.vi" Type="VI" URL="../BeadTracker2.llb/RunSaveImagesExperiment.vi"/>
			<Item Name="SaveExperimentInfo.vi" Type="VI" URL="../BeadTracker2.llb/SaveExperimentInfo.vi"/>
			<Item Name="SendCameraCmd.vi" Type="VI" URL="../BeadTracker2.llb/SendCameraCmd.vi"/>
			<Item Name="SendMotorCmd.vi" Type="VI" URL="../BeadTracker2.llb/SendMotorCmd.vi"/>
			<Item Name="SetMotorAxisPos.vi" Type="VI" URL="../BeadTracker2.llb/SetMotorAxisPos.vi"/>
			<Item Name="StartImageAcquisition.vi" Type="VI" URL="../BeadTracker2.llb/StartImageAcquisition.vi"/>
			<Item Name="StopImageAcquisition.vi" Type="VI" URL="../BeadTracker2.llb/StopImageAcquisition.vi"/>
			<Item Name="StoreFrameInfo.vi" Type="VI" URL="../BeadTracker2.llb/StoreFrameInfo.vi"/>
			<Item Name="TryLoadGainCorrectionImages.vi" Type="VI" URL="../BeadTracker2.llb/TryLoadGainCorrectionImages.vi"/>
			<Item Name="VerifyMeasureConfig.vi" Type="VI" URL="../BeadTracker2.llb/VerifyMeasureConfig.vi"/>
			<Item Name="WriteArrayAsImage.vi" Type="VI" URL="../BeadTracker2.llb/WriteArrayAsImage.vi"/>
			<Item Name="WriteSectionFrame.vi" Type="VI" URL="../BeadTracker2.llb/WriteSectionFrame.vi"/>
		</Item>
		<Item Name="MessageQueue" Type="Folder">
			<Item Name="ListenerType.ctl" Type="VI" URL="../MessageQueue.llb/ListenerType.ctl"/>
			<Item Name="MessageQueue.ctl" Type="VI" URL="../MessageQueue.llb/MessageQueue.ctl"/>
			<Item Name="MessageQueueExample.vi" Type="VI" URL="../MessageQueue.llb/MessageQueueExample.vi"/>
			<Item Name="MessageQueueExample_TestDRV.vi" Type="VI" URL="../MessageQueue.llb/MessageQueueExample_TestDRV.vi"/>
			<Item Name="MessageType.ctl" Type="VI" URL="../MessageQueue.llb/MessageType.ctl"/>
			<Item Name="MsgQueue_Create.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_Create.vi"/>
			<Item Name="MsgQueue_Delete.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_Delete.vi"/>
			<Item Name="MsgQueue_Flush.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_Flush.vi"/>
			<Item Name="MsgQueue_GetNumListeners.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_GetNumListeners.vi"/>
			<Item Name="MsgQueue_GetQueueLengths.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_GetQueueLengths.vi"/>
			<Item Name="MsgQueue_ListenerStatus.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_ListenerStatus.vi"/>
			<Item Name="MsgQueue_ReadMsg.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_ReadMsg.vi"/>
			<Item Name="MsgQueue_RegisterListener.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_RegisterListener.vi"/>
			<Item Name="MsgQueue_RemoveListener.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_RemoveListener.vi"/>
			<Item Name="MsgQueue_SendMsg.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_SendMsg.vi"/>
			<Item Name="MsgQueue_WaitForListeners.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_WaitForListeners.vi"/>
			<Item Name="MsgQueueRef.ctl" Type="VI" URL="../MessageQueue.llb/MsgQueueRef.ctl"/>
		</Item>
		<Item Name="Modules" Type="Folder">
			<Property Name="NI.SortType" Type="Int">3</Property>
			<Item Name="Cameras" Type="Folder">
				<Item Name="VisionExpressCamera.llb" Type="Folder">
					<Item Name="VisionExpressCamera.vi" Type="VI" URL="../Modules/VisionExpressCamera.llb/VisionExpressCamera.vi"/>
				</Item>
			</Item>
			<Item Name="MotorControl" Type="Folder"/>
			<Item Name="FakeCameraAndMotors.vi" Type="VI" URL="../Modules/FakeCameraAndMotors.vi"/>
		</Item>
		<Item Name="QTrk" Type="Folder">
			<Item Name="ResultManager" Type="Folder">
				<Item Name="RMCreate.vi" Type="VI" URL="../qtrk/QTrk.llb/RMCreate.vi"/>
				<Item Name="RMDestroy.vi" Type="VI" URL="../qtrk/QTrk.llb/RMDestroy.vi"/>
				<Item Name="RMFlush.vi" Type="VI" URL="../qtrk/QTrk.llb/RMFlush.vi"/>
				<Item Name="RMGetFrameCounters.vi" Type="VI" URL="../qtrk/QTrk.llb/RMGetFrameCounters.vi"/>
				<Item Name="RMSetTracker.vi" Type="VI" URL="../qtrk/QTrk.llb/RMSetTracker.vi"/>
				<Item Name="RMStoreFrameInfo.vi" Type="VI" URL="../qtrk/QTrk.llb/RMStoreFrameInfo.vi"/>
			</Item>
			<Item Name="QTrkCheckForDLL.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkCheckForDLL.vi"/>
			<Item Name="QTrkClearResults.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkClearResults.vi"/>
			<Item Name="QTrkCreate.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkCreate.vi"/>
			<Item Name="QTrkFlush.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkFlush.vi"/>
			<Item Name="QTrkFree.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkFree.vi"/>
			<Item Name="QTrkFreeAllTrackers.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkFreeAllTrackers.vi"/>
			<Item Name="QTrkGenerateSampleFromLUT.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkGenerateSampleFromLUT.vi"/>
			<Item Name="QTrkGetQueueLength.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkGetQueueLength.vi"/>
			<Item Name="QTrkGetResultCount.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkGetResultCount.vi"/>
			<Item Name="QTrkGetZLUT.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkGetZLUT.vi"/>
			<Item Name="QTrkInstance.ctl" Type="VI" URL="../qtrk/QTrk.llb/QTrkInstance.ctl"/>
			<Item Name="QTrkIsIdle.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkIsIdle.vi"/>
			<Item Name="QTrkLocalizationResult.ctl" Type="VI" URL="../qtrk/QTrk.llb/QTrkLocalizationResult.ctl"/>
			<Item Name="QTrkLocalizationType.ctl" Type="VI" URL="../qtrk/QTrk.llb/QTrkLocalizationType.ctl"/>
			<Item Name="QTrkPixelDataType.ctl" Type="VI" URL="../qtrk/QTrk.llb/QTrkPixelDataType.ctl"/>
			<Item Name="QTrkQueueFrame.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkQueueFrame.vi"/>
			<Item Name="QTrkQueueImageU8.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkQueueImageU8.vi"/>
			<Item Name="QTrkSelectDLLDialog.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkSelectDLLDialog.vi"/>
			<Item Name="QTrkSettings.ctl" Type="VI" URL="../qtrk/QTrk.llb/QTrkSettings.ctl"/>
			<Item Name="QTrkSetZLUT.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkSetZLUT.vi"/>
			<Item Name="QTrkSpeedTest.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkSpeedTest.vi"/>
			<Item Name="QTrkZCommandType.ctl" Type="VI" URL="../qtrk/QTrk.llb/QTrkZCommandType.ctl"/>
		</Item>
		<Item Name="DisplaySpectrum.vi" Type="VI" URL="../Modules/DisplaySpectrum.vi"/>
		<Item Name="MainUI.vi" Type="VI" URL="../BeadTracker2.llb/MainUI.vi"/>
		<Item Name="SetupConfiguration.vi" Type="VI" URL="../SetupConfiguration.vi"/>
		<Item Name="SimpleCameraTest.vi" Type="VI" URL="../SimpleCameraTest.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Application Directory.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Application Directory.vi"/>
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
				<Item Name="IMAQ Attribute.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Attribute.vi"/>
				<Item Name="IMAQ Close.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Close.vi"/>
				<Item Name="IMAQ Configure Buffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Configure Buffer.vi"/>
				<Item Name="IMAQ Configure List.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Configure List.vi"/>
				<Item Name="IMAQ Copy" Type="VI" URL="/&lt;vilib&gt;/vision/Management.llb/IMAQ Copy"/>
				<Item Name="IMAQ Create" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Create"/>
				<Item Name="IMAQ Dispose" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Dispose"/>
				<Item Name="IMAQ Extract Buffer Old Style.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/IMAQ Extract Buffer Old Style.vi"/>
				<Item Name="IMAQ Extract Buffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Extract Buffer.vi"/>
				<Item Name="IMAQ GetImagePixelPtr" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ GetImagePixelPtr"/>
				<Item Name="IMAQ GetImageSize" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ GetImageSize"/>
				<Item Name="IMAQ Image.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/IMAQ Image.ctl"/>
				<Item Name="IMAQ ImageToArray" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ImageToArray"/>
				<Item Name="IMAQ Init.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Init.vi"/>
				<Item Name="IMAQ Start.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Start.vi"/>
				<Item Name="IMAQ Write BMP File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write BMP File 2"/>
				<Item Name="IMAQ Write File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write File 2"/>
				<Item Name="IMAQ Write Image And Vision Info File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write Image And Vision Info File 2"/>
				<Item Name="IMAQ Write JPEG File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write JPEG File 2"/>
				<Item Name="IMAQ Write JPEG2000 File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write JPEG2000 File 2"/>
				<Item Name="IMAQ Write PNG File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write PNG File 2"/>
				<Item Name="IMAQ Write TIFF File 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Write TIFF File 2"/>
				<Item Name="IMAQRegisterSession.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/IMAQRegisterSession.vi"/>
				<Item Name="IMAQUnregisterSession.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/IMAQUnregisterSession.vi"/>
				<Item Name="imgBufferElement.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgBufferElement.vi"/>
				<Item Name="imgClose.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgClose.vi"/>
				<Item Name="imgCreateBufList.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgCreateBufList.vi"/>
				<Item Name="imgDisposeBufList.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgDisposeBufList.vi"/>
				<Item Name="imgEnsureEqualBorders.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgEnsureEqualBorders.vi"/>
				<Item Name="imgGetBufList.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgGetBufList.vi"/>
				<Item Name="imgInterfaceOpen.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgInterfaceOpen.vi"/>
				<Item Name="imgIsNewStyleInterface.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgIsNewStyleInterface.vi"/>
				<Item Name="imgMemLock.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgMemLock.vi"/>
				<Item Name="imgSessionAcquire.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionAcquire.vi"/>
				<Item Name="imgSessionAttribute.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionAttribute.vi"/>
				<Item Name="imgSessionConfigure.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionConfigure.vi"/>
				<Item Name="imgSessionExamineBuffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionExamineBuffer.vi"/>
				<Item Name="imgSessionOpen.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionOpen.vi"/>
				<Item Name="imgSessionReleaseBuffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionReleaseBuffer.vi"/>
				<Item Name="imgSetRoi.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSetRoi.vi"/>
				<Item Name="imgUpdateErrorCluster.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgUpdateErrorCluster.vi"/>
				<Item Name="imgWaitForIMAQOccurrence.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgWaitForIMAQOccurrence.vi"/>
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
				<Item Name="SessionLookUp.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/SessionLookUp.vi"/>
				<Item Name="Set Bold Text.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set Bold Text.vi"/>
				<Item Name="Set String Value.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set String Value.vi"/>
				<Item Name="subFile Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/express/express input/FileDialogBlock.llb/subFile Dialog.vi"/>
				<Item Name="TagReturnType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/TagReturnType.ctl"/>
				<Item Name="Three Button Dialog CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog CORE.vi"/>
				<Item Name="Three Button Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog.vi"/>
				<Item Name="Trim Whitespace.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Trim Whitespace.vi"/>
				<Item Name="Vision Acquisition CalculateFPS.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/Vision Acquisition Express Utility VIs.llb/Vision Acquisition CalculateFPS.vi"/>
				<Item Name="Vision Acquisition IMAQ Filter Stop Trigger Error.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/Vision Acquisition Express Utility VIs.llb/Vision Acquisition IMAQ Filter Stop Trigger Error.vi"/>
				<Item Name="whitespace.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/whitespace.ctl"/>
				<Item Name="Write Spreadsheet String.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write Spreadsheet String.vi"/>
				<Item Name="Write To Spreadsheet File (DBL).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File (DBL).vi"/>
				<Item Name="Write To Spreadsheet File (I64).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File (I64).vi"/>
				<Item Name="Write To Spreadsheet File (string).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File (string).vi"/>
				<Item Name="Write To Spreadsheet File.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File.vi"/>
			</Item>
			<Item Name="CmdEnum_CameraOut.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_CameraOut.ctl"/>
			<Item Name="CreateFileDirectory.vi" Type="VI" URL="../BeadTracker2.llb/CreateFileDirectory.vi"/>
			<Item Name="GetSetMotorAxisValue.vi" Type="VI" URL="../BeadTracker2.llb/GetSetMotorAxisValue.vi"/>
			<Item Name="imaq.dll" Type="Document" URL="imaq.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="kernel32.dll" Type="Document" URL="kernel32.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="lvanlys.dll" Type="Document" URL="/C/Program Files (x86)/National Instruments/LabVIEW 2011/resource/lvanlys.dll"/>
			<Item Name="MakeStetsonWindow.vi" Type="VI" URL="../qtrk/QTrk.llb/MakeStetsonWindow.vi"/>
			<Item Name="nivision.dll" Type="Document" URL="nivision.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="NotifyUI.vi" Type="VI" URL="../BeadTracker2.llb/NotifyUI.vi"/>
			<Item Name="QTrkAccurateTickCount.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkAccurateTickCount.vi"/>
			<Item Name="QTrkComputedSettings.ctl" Type="VI" URL="../qtrk/QTrk.llb/QTrkComputedSettings.ctl"/>
			<Item Name="QTrkComputeLUTFisherMatrix.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkComputeLUTFisherMatrix.vi"/>
			<Item Name="QTrkDLL.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkDLL.vi"/>
			<Item Name="QTrkGetDebugImage.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkGetDebugImage.vi"/>
			<Item Name="QTrkGetDerivedSettings.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkGetDerivedSettings.vi"/>
			<Item Name="QTrkGetProfilingReport.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkGetProfilingReport.vi"/>
			<Item Name="QTrkLocalizationJob.ctl" Type="VI" URL="../qtrk/QTrk.llb/QTrkLocalizationJob.ctl"/>
			<Item Name="QTrkSetPixelGainOffset.vi" Type="VI" URL="../qtrk/QTrk.llb/QTrkSetPixelGainOffset.vi"/>
			<Item Name="ResultManagerConfig.ctl" Type="VI" URL="../qtrk/QTrk.llb/ResultManagerConfig.ctl"/>
			<Item Name="ResultManagerInstance.ctl" Type="VI" URL="../qtrk/QTrk.llb/ResultManagerInstance.ctl"/>
			<Item Name="RMDiscardBead.vi" Type="VI" URL="../qtrk/QTrk.llb/RMDiscardBead.vi"/>
			<Item Name="RMGetBeadResults.vi" Type="VI" URL="../qtrk/QTrk.llb/RMGetBeadResults.vi"/>
			<Item Name="SelectBeads.vi" Type="VI" URL="../BeadTracker2.llb/SelectBeads.vi"/>
			<Item Name="UserInterfaceEventType.ctl" Type="VI" URL="../BeadTracker2.llb/UserInterfaceEventType.ctl"/>
			<Item Name="XYZf.ctl" Type="VI" URL="../qtrk/QTrk.llb/XYZf.ctl"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
