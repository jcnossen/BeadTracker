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
			<Item Name="LoadOneImageviaPath.vi" Type="VI" URL="../AutoBeadFinder.llb/LoadOneImageviaPath.vi"/>
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
				<Item Name="ExperimentStateType.ctl" Type="VI" URL="../BeadTracker2.llb/ExperimentStateType.ctl"/>
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
				<Item Name="AdvanceFramesAllDone.vi" Type="VI" URL="../BeadTracker2.llb/AdvanceFramesAllDone.vi"/>
				<Item Name="AllocateMemoryForResults.vi" Type="VI" URL="../BeadTracker2.llb/AllocateMemoryForResults.vi"/>
				<Item Name="BuildZLUT.vi" Type="VI" URL="../BeadTracker2.llb/BuildZLUT.vi"/>
				<Item Name="CreateQTrkInstance.vi" Type="VI" URL="../BeadTracker2.llb/CreateQTrkInstance.vi"/>
				<Item Name="CreateResultDVR.vi" Type="VI" URL="../BeadTracker2.llb/CreateResultDVR.vi"/>
				<Item Name="DiscardBead.vi" Type="VI" URL="../BeadTracker2.llb/DiscardBead.vi"/>
				<Item Name="EditQTrkSettingsDialog.vi" Type="VI" URL="../BeadTracker2.llb/EditQTrkSettingsDialog.vi"/>
				<Item Name="FetchTrackingResults.vi" Type="VI" URL="../BeadTracker2.llb/FetchTrackingResults.vi"/>
				<Item Name="GetBeadCornerPos.vi" Type="VI" URL="../BeadTracker2.llb/GetBeadCornerPos.vi"/>
				<Item Name="GetResultScaleAndOffset.vi" Type="VI" URL="../BeadTracker2.llb/GetResultScaleAndOffset.vi"/>
				<Item Name="PartialFreeResultData.vi" Type="VI" URL="../BeadTracker2.llb/PartialFreeResultData.vi"/>
				<Item Name="ResultsToXYZGraphData.vi" Type="VI" URL="../BeadTracker2.llb/ResultsToXYZGraphData.vi"/>
				<Item Name="SaveOrLoadBeadlist.vi" Type="VI" URL="../BeadTracker2.llb/SaveOrLoadBeadlist.vi"/>
				<Item Name="SaveTrackingResults.vi" Type="VI" URL="../BeadTracker2.llb/SaveTrackingResults.vi"/>
			</Item>
			<Item Name="Typedefs" Type="Folder">
				<Item Name="Cmd_CameraIn.ctl" Type="VI" URL="../BeadTracker2.llb/Cmd_CameraIn.ctl"/>
				<Item Name="Cmd_MotorIn.ctl" Type="VI" URL="../BeadTracker2.llb/Cmd_MotorIn.ctl"/>
				<Item Name="CmdData_NewFrame.ctl" Type="VI" URL="../BeadTracker2.llb/CmdData_NewFrame.ctl"/>
				<Item Name="CmdData_SetMotorPos.ctl" Type="VI" URL="../BeadTracker2.llb/CmdData_SetMotorPos.ctl"/>
				<Item Name="CmdEnum_CameraIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_CameraIn.ctl"/>
				<Item Name="CmdEnum_MotorIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_MotorIn.ctl"/>
				<Item Name="CmdEnum_UserInterfaceCmd.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_UserInterfaceCmd.ctl"/>
				<Item Name="FrameInfoType.ctl" Type="VI" URL="../BeadTracker2.llb/FrameInfoType.ctl"/>
				<Item Name="MeasureConfigType.ctl" Type="VI" URL="../BeadTracker2.llb/MeasureConfigType.ctl"/>
				<Item Name="MotorConfigType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorConfigType.ctl"/>
				<Item Name="MotorPosFlagsType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorPosFlagsType.ctl"/>
				<Item Name="MotorPosType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorPosType.ctl"/>
				<Item Name="MotorStateType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorStateType.ctl"/>
				<Item Name="QueueListType.ctl" Type="VI" URL="../BeadTracker2.llb/QueueListType.ctl"/>
				<Item Name="ResultsCollectionType.ctl" Type="VI" URL="../BeadTracker2.llb/ResultsCollectionType.ctl"/>
				<Item Name="TrackedBeadPos.ctl" Type="VI" URL="../BeadTracker2.llb/TrackedBeadPos.ctl"/>
				<Item Name="XYf.ctl" Type="VI" URL="../BeadTracker2.llb/XYf.ctl"/>
			</Item>
			<Item Name="AskForCreateNewDir.vi" Type="VI" URL="../BeadTracker2.llb/AskForCreateNewDir.vi"/>
			<Item Name="ClusterToValueList.vi" Type="VI" URL="../BeadTracker2.llb/ClusterToValueList.vi"/>
			<Item Name="ComputeHighFreqNoise.vi" Type="VI" URL="../BeadTracker2.llb/ComputeHighFreqNoise.vi"/>
			<Item Name="DeleteFileIfExisting.vi" Type="VI" URL="../BeadTracker2.llb/DeleteFileIfExisting.vi"/>
			<Item Name="GenerateNewFilename.vi" Type="VI" URL="../BeadTracker2.llb/GenerateNewFilename.vi"/>
			<Item Name="GetExperimentPaths.vi" Type="VI" URL="../BeadTracker2.llb/GetExperimentPaths.vi"/>
			<Item Name="GetQueues.vi" Type="VI" URL="../BeadTracker2.llb/GetQueues.vi"/>
			<Item Name="GlobalVariables.vi" Type="VI" URL="../BeadTracker2.llb/GlobalVariables.vi"/>
			<Item Name="LimitedMotorMoveCmd.vi" Type="VI" URL="../BeadTracker2.llb/LimitedMotorMoveCmd.vi"/>
			<Item Name="LimitMotorPos.vi" Type="VI" URL="../BeadTracker2.llb/LimitMotorPos.vi"/>
			<Item Name="MotorUI.vi" Type="VI" URL="../BeadTracker2.llb/MotorUI.vi"/>
			<Item Name="OpenOrContinueResultFile.vi" Type="VI" URL="../BeadTracker2.llb/OpenOrContinueResultFile.vi"/>
			<Item Name="ResultsToTracePlotData.vi" Type="VI" URL="../BeadTracker2.llb/ResultsToTracePlotData.vi"/>
			<Item Name="SaveExperimentInfo.vi" Type="VI" URL="../BeadTracker2.llb/SaveExperimentInfo.vi"/>
			<Item Name="SendCameraCmd.vi" Type="VI" URL="../BeadTracker2.llb/SendCameraCmd.vi"/>
			<Item Name="SendMotorCmd.vi" Type="VI" URL="../BeadTracker2.llb/SendMotorCmd.vi"/>
			<Item Name="SetMotorAxisPos.vi" Type="VI" URL="../BeadTracker2.llb/SetMotorAxisPos.vi"/>
			<Item Name="ShowRenameIfExistingExpDialog.vi" Type="VI" URL="../BeadTracker2.llb/ShowRenameIfExistingExpDialog.vi"/>
			<Item Name="TrackerMain.vi" Type="VI" URL="../BeadTracker2.llb/TrackerMain.vi"/>
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
		</Item>
		<Item Name="QTrk" Type="Folder">
			<Item Name="QTrkCheckForDLL.vi" Type="VI" URL="../QTrk.llb/QTrkCheckForDLL.vi"/>
			<Item Name="QTrkClearResults.vi" Type="VI" URL="../QTrk.llb/QTrkClearResults.vi"/>
			<Item Name="QTrkCreate.vi" Type="VI" URL="../QTrk.llb/QTrkCreate.vi"/>
			<Item Name="QTrkDLL.vi" Type="VI" URL="../QTrk.llb/QTrkDLL.vi"/>
			<Item Name="QTrkFlush.vi" Type="VI" URL="../QTrk.llb/QTrkFlush.vi"/>
			<Item Name="QTrkFree.vi" Type="VI" URL="../QTrk.llb/QTrkFree.vi"/>
			<Item Name="QTrkFreeAllTrackers.vi" Type="VI" URL="../QTrk.llb/QTrkFreeAllTrackers.vi"/>
			<Item Name="QTrkGenerateSampleFromLUT.vi" Type="VI" URL="../QTrk.llb/QTrkGenerateSampleFromLUT.vi"/>
			<Item Name="QTrkGetQueueSize.vi" Type="VI" URL="../QTrk.llb/QTrkGetQueueSize.vi"/>
			<Item Name="QTrkGetResultCount.vi" Type="VI" URL="../QTrk.llb/QTrkGetResultCount.vi"/>
			<Item Name="QTrkGetResults.vi" Type="VI" URL="../QTrk.llb/QTrkGetResults.vi"/>
			<Item Name="QTrkGetSingleResult.vi" Type="VI" URL="../QTrk.llb/QTrkGetSingleResult.vi"/>
			<Item Name="QTrkGetZLUT.vi" Type="VI" URL="../QTrk.llb/QTrkGetZLUT.vi"/>
			<Item Name="QTrkInstance.ctl" Type="VI" URL="../QTrk.llb/QTrkInstance.ctl"/>
			<Item Name="QTrkIsIdle.vi" Type="VI" URL="../QTrk.llb/QTrkIsIdle.vi"/>
			<Item Name="QTrkIsQueueFull.vi" Type="VI" URL="../QTrk.llb/QTrkIsQueueFull.vi"/>
			<Item Name="QTrkLocalizationResult.ctl" Type="VI" URL="../QTrk.llb/QTrkLocalizationResult.ctl"/>
			<Item Name="QTrkPixelDataType.ctl" Type="VI" URL="../QTrk.llb/QTrkPixelDataType.ctl"/>
			<Item Name="QTrkQueueFrame.vi" Type="VI" URL="../QTrk.llb/QTrkQueueFrame.vi"/>
			<Item Name="QTrkQueueImageU8.vi" Type="VI" URL="../QTrk.llb/QTrkQueueImageU8.vi"/>
			<Item Name="QTrkQueueImageU16.vi" Type="VI" URL="../QTrk.llb/QTrkQueueImageU16.vi"/>
			<Item Name="QTrkReadJPEGFile.vi" Type="VI" URL="../QTrk.llb/QTrkReadJPEGFile.vi"/>
			<Item Name="QTrkSelectDLLDialog.vi" Type="VI" URL="../QTrk.llb/QTrkSelectDLLDialog.vi"/>
			<Item Name="QTrkSettings.ctl" Type="VI" URL="../QTrk.llb/QTrkSettings.ctl"/>
			<Item Name="QTrkSetZLUT.vi" Type="VI" URL="../QTrk.llb/QTrkSetZLUT.vi"/>
			<Item Name="QTrkWaitForResults.vi" Type="VI" URL="../QTrk.llb/QTrkWaitForResults.vi"/>
		</Item>
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
				<Item Name="IMAQ Create" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Create"/>
				<Item Name="IMAQ Extract Buffer Old Style.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/IMAQ Extract Buffer Old Style.vi"/>
				<Item Name="IMAQ Extract Buffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Extract Buffer.vi"/>
				<Item Name="IMAQ GetImagePixelPtr" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ GetImagePixelPtr"/>
				<Item Name="IMAQ GetImageSize" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ GetImageSize"/>
				<Item Name="IMAQ Image.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/IMAQ Image.ctl"/>
				<Item Name="IMAQ ImageToArray" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ImageToArray"/>
				<Item Name="IMAQ Init.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Init.vi"/>
				<Item Name="IMAQ ReadFile" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ ReadFile"/>
				<Item Name="IMAQ Serial Read.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Serial Read.vi"/>
				<Item Name="IMAQ Serial Write.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Serial Write.vi"/>
				<Item Name="IMAQ Start.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Start.vi"/>
				<Item Name="IMAQ Status.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Status.vi"/>
				<Item Name="IMAQ Stop.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Stop.vi"/>
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
				<Item Name="imgSessionStatus.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionStatus.vi"/>
				<Item Name="imgSessionStopAcquisition.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSessionStopAcquisition.vi"/>
				<Item Name="imgSetRoi.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgSetRoi.vi"/>
				<Item Name="imgUpdateErrorCluster.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgUpdateErrorCluster.vi"/>
				<Item Name="imgWaitForIMAQOccurrence.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/imgWaitForIMAQOccurrence.vi"/>
				<Item Name="Longest Line Length in Pixels.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Longest Line Length in Pixels.vi"/>
				<Item Name="LVBoundsTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVBoundsTypeDef.ctl"/>
				<Item Name="LVPointDoubleTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVPointDoubleTypeDef.ctl"/>
				<Item Name="NI_AALBase.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALBase.lvlib"/>
				<Item Name="NI_AALPro.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALPro.lvlib"/>
				<Item Name="NI_FileType.lvlib" Type="Library" URL="/&lt;vilib&gt;/Utility/lvfile.llb/NI_FileType.lvlib"/>
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
				<Item Name="System Exec.vi" Type="VI" URL="/&lt;vilib&gt;/Platform/system.llb/System Exec.vi"/>
				<Item Name="TagReturnType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/TagReturnType.ctl"/>
				<Item Name="Three Button Dialog CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog CORE.vi"/>
				<Item Name="Three Button Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog.vi"/>
				<Item Name="Trim Whitespace.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Trim Whitespace.vi"/>
				<Item Name="VISA Configure Serial Port" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port"/>
				<Item Name="VISA Configure Serial Port (Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Instr).vi"/>
				<Item Name="VISA Configure Serial Port (Serial Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Serial Instr).vi"/>
				<Item Name="Vision Acquisition CalculateFPS.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/Vision Acquisition Express Utility VIs.llb/Vision Acquisition CalculateFPS.vi"/>
				<Item Name="Vision Acquisition IMAQ Filter Stop Trigger Error.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/Vision Acquisition Express Utility VIs.llb/Vision Acquisition IMAQ Filter Stop Trigger Error.vi"/>
				<Item Name="whitespace.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/whitespace.ctl"/>
				<Item Name="Write Spreadsheet String.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write Spreadsheet String.vi"/>
				<Item Name="Write To Spreadsheet File (DBL).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File (DBL).vi"/>
				<Item Name="Write To Spreadsheet File (I64).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File (I64).vi"/>
				<Item Name="Write To Spreadsheet File (string).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File (string).vi"/>
				<Item Name="Write To Spreadsheet File.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write To Spreadsheet File.vi"/>
			</Item>
			<Item Name="#5.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Special command.llb/#5.vi"/>
			<Item Name="#5_old.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Old commands.llb/#5_old.vi"/>
			<Item Name="#7.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Special command.llb/#7.vi"/>
			<Item Name="#9.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/WaveGenerator.llb/#9.vi"/>
			<Item Name="#24.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Special command.llb/#24.vi"/>
			<Item Name="*IDN?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/*IDN?.vi"/>
			<Item Name="AccurateTickCount.vi" Type="VI" URL="../BeadTracker2.llb/AccurateTickCount.vi"/>
			<Item Name="Analog FGlobal.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Analog control.llb/Analog FGlobal.vi"/>
			<Item Name="Analog Functions.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Analog control.llb/Analog Functions.vi"/>
			<Item Name="Analog Receive String.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Analog control.llb/Analog Receive String.vi"/>
			<Item Name="Assign booleans from string to axes.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Assign booleans from string to axes.vi"/>
			<Item Name="Assign NaN for chosen axes.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Assign NaN for chosen axes.vi"/>
			<Item Name="Assign values from string to axes.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Assign values from string to axes.vi"/>
			<Item Name="ATZ.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/ATZ.vi"/>
			<Item Name="ATZ?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/ATZ?.vi"/>
			<Item Name="Available Analog Commands.ctl" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Analog control.llb/Available Analog Commands.ctl"/>
			<Item Name="Available DLL interfaces.ctl" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Communication.llb/Available DLL interfaces.ctl"/>
			<Item Name="Available DLLs.ctl" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Communication.llb/Available DLLs.ctl"/>
			<Item Name="Available interfaces.ctl" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Communication.llb/Available interfaces.ctl"/>
			<Item Name="BT2_CameraModule.vi" Type="VI" URL="../Modules/FastCMOS.llb/BT2_CameraModule.vi"/>
			<Item Name="Build command substring.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Build command substring.vi"/>
			<Item Name="Build query command substring.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Support.llb/Build query command substring.vi"/>
			<Item Name="CameraConfig.ctl" Type="VI" URL="../Modules/FastCMOS.llb/CameraConfig.ctl"/>
			<Item Name="Close connection if open.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/Close connection if open.vi"/>
			<Item Name="Cmd_SetPosition.vi" Type="VI" URL="../Setups/D020L/PIMotorController.llb/Cmd_SetPosition.vi"/>
			<Item Name="Cmd_TrackerIn.ctl" Type="VI" URL="../BeadTracker2.llb/Cmd_TrackerIn.ctl"/>
			<Item Name="CmdEnum_CameraOut.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_CameraOut.ctl"/>
			<Item Name="Combine axes arrays.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Combine axes arrays.vi"/>
			<Item Name="Commanded axes connected?.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Support.llb/Commanded axes connected?.vi"/>
			<Item Name="Controller names.ctl" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/General command.llb/Controller names.ctl"/>
			<Item Name="CreateFileDirectory.vi" Type="VI" URL="../BeadTracker2.llb/CreateFileDirectory.vi"/>
			<Item Name="CST?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Special command.llb/CST?.vi"/>
			<Item Name="Cut out additional spaces.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Support.llb/Cut out additional spaces.vi"/>
			<Item Name="D020L_Motors.vi" Type="VI" URL="../Setups/D020L/D020L_Motors.vi"/>
			<Item Name="Define axes to command from boolean array.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Define axes to command from boolean array.vi"/>
			<Item Name="Define connected axes.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/Define connected axes.vi"/>
			<Item Name="Define connected systems (Array).vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/Define connected systems (Array).vi"/>
			<Item Name="E712_Configuration_Setup.vi" Type="VI" URL="../Setups/D020L/MotorsInit.llb/E712_Configuration_Setup.vi"/>
			<Item Name="ERR?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/ERR?.vi"/>
			<Item Name="Find host address.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/Find host address.vi"/>
			<Item Name="FNL.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/FNL.vi"/>
			<Item Name="FPL.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/FPL.vi"/>
			<Item Name="FRF.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/FRF.vi"/>
			<Item Name="FRF?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/FRF?.vi"/>
			<Item Name="GCSTranslateError.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/GCSTranslateError.vi"/>
			<Item Name="GCSTranslator DLL Functions.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Communication.llb/GCSTranslator DLL Functions.vi"/>
			<Item Name="GCSTranslator.dll" Type="Document" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/GCSTranslator.dll"/>
			<Item Name="General wait for movement to stop.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/General wait for movement to stop.vi"/>
			<Item Name="Get all axes.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Support.llb/Get all axes.vi"/>
			<Item Name="Get arrays without blanks.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Support.llb/Get arrays without blanks.vi"/>
			<Item Name="Get lines from string.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Support.llb/Get lines from string.vi"/>
			<Item Name="Get subnet.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/Get subnet.vi"/>
			<Item Name="GetSetMotorAxisValue.vi" Type="VI" URL="../BeadTracker2.llb/GetSetMotorAxisValue.vi"/>
			<Item Name="GetSingleAxisPos.vi" Type="VI" URL="../Setups/D020L/PIMotorController.llb/GetSingleAxisPos.vi"/>
			<Item Name="Global Analog.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Analog control.llb/Global Analog.vi"/>
			<Item Name="Global DaisyChain.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Communication.llb/Global DaisyChain.vi"/>
			<Item Name="Global1.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Communication.llb/Global1.vi"/>
			<Item Name="Global2 (Array).vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/General command.llb/Global2 (Array).vi"/>
			<Item Name="HLP?.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/General command.llb/HLP?.vi"/>
			<Item Name="imaq.dll" Type="Document" URL="imaq.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="Initialize Global DaisyChain.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/Initialize Global DaisyChain.vi"/>
			<Item Name="Initialize Global1.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/Initialize Global1.vi"/>
			<Item Name="Initialize Global2.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/Initialize Global2.vi"/>
			<Item Name="Is DaisyChain open.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/Is DaisyChain open.vi"/>
			<Item Name="kernel32.dll" Type="Document" URL="kernel32.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="LIM?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/LIM?.vi"/>
			<Item Name="Longlasting one-axis command.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Longlasting one-axis command.vi"/>
			<Item Name="lvanlys.dll" Type="Document" URL="/C/Program Files/National Instruments/LabVIEW 2011/resource/lvanlys.dll"/>
			<Item Name="MakeStetsonWindow.vi" Type="VI" URL="../QTrk.llb/MakeStetsonWindow.vi"/>
			<Item Name="MeasureCurrentPos.vi" Type="VI" URL="../Setups/D020L/PIMotorController.llb/MeasureCurrentPos.vi"/>
			<Item Name="Mercury_GCS_Configuration_Setup.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Mercury_GCS_Configuration_Setup.vi"/>
			<Item Name="mkConfigure.vi" Type="VI" URL="../Modules/FastCMOS.llb/mkConfigure.vi"/>
			<Item Name="mkConfigureBufferList.vi" Type="VI" URL="../Modules/FastCMOS.llb/mkConfigureBufferList.vi"/>
			<Item Name="mkGetROIs.vi" Type="VI" URL="../Modules/FastCMOS.llb/mkGetROIs.vi"/>
			<Item Name="mkGetSetFramerate.vi" Type="VI" URL="../Modules/FastCMOS.llb/mkGetSetFramerate.vi"/>
			<Item Name="mkGetSetInFrameCounter.vi" Type="VI" URL="../Modules/FastCMOS.llb/mkGetSetInFrameCounter.vi"/>
			<Item Name="mkSendSerialCmd.vi" Type="VI" URL="../Modules/FastCMOS.llb/mkSendSerialCmd.vi"/>
			<Item Name="mkSetExposureGainOffset.vi" Type="VI" URL="../Modules/FastCMOS.llb/mkSetExposureGainOffset.vi"/>
			<Item Name="mkSetMode.vi" Type="VI" URL="../Modules/FastCMOS.llb/mkSetMode.vi"/>
			<Item Name="mkSetROIs.vi" Type="VI" URL="../Modules/FastCMOS.llb/mkSetROIs.vi"/>
			<Item Name="Motors_init_all_D020.vi" Type="VI" URL="../Setups/D020L/MotorsInit.llb/Motors_init_all_D020.vi"/>
			<Item Name="Motors_Init_E712_D020.vi" Type="VI" URL="../Setups/D020L/MotorsInit.llb/Motors_Init_E712_D020.vi"/>
			<Item Name="Motors_Mercury_init_D012_MD.vi" Type="VI" URL="../Setups/D020L/MotorsInit.llb/Motors_Mercury_init_D012_MD.vi"/>
			<Item Name="MOV.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/MOV.vi"/>
			<Item Name="nivision.dll" Type="Document" URL="nivision.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="ONT?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/ONT?.vi"/>
			<Item Name="PI Open Interface of one system.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/PI Open Interface of one system.vi"/>
			<Item Name="PI Receive String.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/PI Receive String.vi"/>
			<Item Name="PI Send String.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Communication.llb/PI Send String.vi"/>
			<Item Name="PI VISA Receive Characters.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/PI VISA Receive Characters.vi"/>
			<Item Name="PI_Axis.ctl" Type="VI" URL="../Setups/D020L/PIMotorController.llb/PI_Axis.ctl"/>
			<Item Name="PI_Stages_Main.vi" Type="VI" URL="../Setups/D020L/PIMotorController.llb/PI_Stages_Main.vi"/>
			<Item Name="PIAxisInfoGlobal.vi" Type="VI" URL="../Setups/D020L/PIMotorController.llb/PIAxisInfoGlobal.vi"/>
			<Item Name="POS?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/POS?.vi"/>
			<Item Name="QTrkLocalizationJob.ctl" Type="VI" URL="../QTrk.llb/QTrkLocalizationJob.ctl"/>
			<Item Name="Return space.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Support.llb/Return space.vi"/>
			<Item Name="RON.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/RON.vi"/>
			<Item Name="RON?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/RON?.vi"/>
			<Item Name="SAI?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/SAI?.vi"/>
			<Item Name="Select DaisyChain device.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/Select DaisyChain device.vi"/>
			<Item Name="Select host address.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/Select host address.vi"/>
			<Item Name="Select USB device.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Communication.llb/Select USB device.vi"/>
			<Item Name="Select values for chosen axes.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Select values for chosen axes.vi"/>
			<Item Name="SelectBeads.vi" Type="VI" URL="../BeadTracker2.llb/SelectBeads.vi"/>
			<Item Name="Set RON and return RON status.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Set RON and return RON status.vi"/>
			<Item Name="SetSingleAxisPos.vi" Type="VI" URL="../Setups/D020L/PIMotorController.llb/SetSingleAxisPos.vi"/>
			<Item Name="ShowSettingsDialog.vi" Type="VI" URL="../Modules/FastCMOS.llb/ShowSettingsDialog.vi"/>
			<Item Name="STA?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Special command.llb/STA?.vi"/>
			<Item Name="String with ASCII code conversion.vi" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Support.llb/String with ASCII code conversion.vi"/>
			<Item Name="Substract axes array subset from axes array.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Substract axes array subset from axes array.vi"/>
			<Item Name="SVO.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/SVO.vi"/>
			<Item Name="SVO?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/General command.llb/SVO?.vi"/>
			<Item Name="Termination character.ctl" Type="VI" URL="/C/Users/Public/Documents/MercuryGCS/GCS_LabView/Low Level/Communication.llb/Termination character.ctl"/>
			<Item Name="TMN?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/TMN?.vi"/>
			<Item Name="TMX?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/TMX?.vi"/>
			<Item Name="TRS?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Limits.llb/TRS?.vi"/>
			<Item Name="TWG?.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/WaveGenerator.llb/TWG?.vi"/>
			<Item Name="Wait for answer of longlasting command.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Wait for answer of longlasting command.vi"/>
			<Item Name="Wait for axes to stop.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Wait for axes to stop.vi"/>
			<Item Name="Wait for controller ready.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Support.llb/Wait for controller ready.vi"/>
			<Item Name="Wait for hexapod system axes to stop.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/Old commands.llb/Wait for hexapod system axes to stop.vi"/>
			<Item Name="WGO.vi" Type="VI" URL="/C/Users/Public/Documents/Merged_GCS_LabVIEW/Merged_GCS_LabVIEW/Low Level/WaveGenerator.llb/WGO.vi"/>
			<Item Name="XYZf.ctl" Type="VI" URL="../QTrk.llb/XYZf.ctl"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
