<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="11008008">
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
			<Item Name="Typedefs" Type="Folder">
				<Item Name="CameraStateType.ctl" Type="VI" URL="../BeadTracker2.llb/CameraStateType.ctl"/>
				<Item Name="CmdData_NewFrame.ctl" Type="VI" URL="../BeadTracker2.llb/CmdData_NewFrame.ctl"/>
				<Item Name="CmdData_SetMotorPos.ctl" Type="VI" URL="../BeadTracker2.llb/CmdData_SetMotorPos.ctl"/>
				<Item Name="CmdEnum_CameraIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_CameraIn.ctl"/>
				<Item Name="CmdEnum_MotorIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_MotorIn.ctl"/>
				<Item Name="CmdType_CameraIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdType_CameraIn.ctl"/>
				<Item Name="CmdType_CameraOut.ctl" Type="VI" URL="../BeadTracker2.llb/CmdType_CameraOut.ctl"/>
				<Item Name="CmdType_MotorIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdType_MotorIn.ctl"/>
				<Item Name="MeasureConfigType.ctl" Type="VI" URL="../BeadTracker2.llb/MeasureConfigType.ctl"/>
				<Item Name="MotorMoveFlagsType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorMoveFlagsType.ctl"/>
				<Item Name="MotorPosType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorPosType.ctl"/>
				<Item Name="MotorStateType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorStateType.ctl"/>
				<Item Name="QueueListType.ctl" Type="VI" URL="../BeadTracker2.llb/QueueListType.ctl"/>
				<Item Name="XYf.ctl" Type="VI" URL="../BeadTracker2.llb/XYf.ctl"/>
			</Item>
			<Item Name="AskForCreateNewDir.vi" Type="VI" URL="../BeadTracker2.llb/AskForCreateNewDir.vi"/>
			<Item Name="BuildZLUT.vi" Type="VI" URL="../BeadTracker2.llb/BuildZLUT.vi"/>
			<Item Name="EditQTrkSettingsDialog.vi" Type="VI" URL="../BeadTracker2.llb/EditQTrkSettingsDialog.vi"/>
			<Item Name="GetExperimentPaths.vi" Type="VI" URL="../BeadTracker2.llb/GetExperimentPaths.vi"/>
			<Item Name="GetQueues.vi" Type="VI" URL="../BeadTracker2.llb/GetQueues.vi"/>
			<Item Name="GlobalVariables.vi" Type="VI" URL="../BeadTracker2.llb/GlobalVariables.vi"/>
			<Item Name="GrabImageFromQueue.vi" Type="VI" URL="../BeadTracker2.llb/GrabImageFromQueue.vi"/>
			<Item Name="MotorUI.vi" Type="VI" URL="../BeadTracker2.llb/MotorUI.vi"/>
			<Item Name="MoveMotorAxis.vi" Type="VI" URL="../BeadTracker2.llb/MoveMotorAxis.vi"/>
			<Item Name="SaveOrLoadBeadlist.vi" Type="VI" URL="../BeadTracker2.llb/SaveOrLoadBeadlist.vi"/>
			<Item Name="SendCameraCmd.vi" Type="VI" URL="../BeadTracker2.llb/SendCameraCmd.vi"/>
			<Item Name="SendMotorCmd.vi" Type="VI" URL="../BeadTracker2.llb/SendMotorCmd.vi"/>
			<Item Name="SendMotorMoveCmd.vi" Type="VI" URL="../BeadTracker2.llb/SendMotorMoveCmd.vi"/>
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
			<Item Name="MsgQueue_ReadMsg.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_ReadMsg.vi"/>
			<Item Name="MsgQueue_RegisterListener.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_RegisterListener.vi"/>
			<Item Name="MsgQueue_RemoveListener.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_RemoveListener.vi"/>
			<Item Name="MsgQueue_SendMsg.vi" Type="VI" URL="../MessageQueue.llb/MsgQueue_SendMsg.vi"/>
			<Item Name="MsgQueueRef.ctl" Type="VI" URL="../MessageQueue.llb/MsgQueueRef.ctl"/>
		</Item>
		<Item Name="Modules" Type="Folder">
			<Property Name="NI.SortType" Type="Int">3</Property>
			<Item Name="Cameras" Type="Folder">
				<Item Name="VisionExpressCamera.llb" Type="Folder">
					<Item Name="VisionExpressCamera.vi" Type="VI" URL="../Modules/VisionExpressCamera.llb/VisionExpressCamera.vi"/>
				</Item>
			</Item>
			<Item Name="MotorControl" Type="Folder">
				<Item Name="PI_M126_E816Piezo.llb" Type="Folder">
					<Item Name="MeasureCurrentPos.vi" Type="VI" URL="../Modules/PI_M126_E816Piezo.llb/MeasureCurrentPos.vi"/>
					<Item Name="Cmd_SetPosition.vi" Type="VI" URL="../Modules/PI_M126_E816Piezo.llb/Cmd_SetPosition.vi"/>
					<Item Name="PI_Stages_Main.vi" Type="VI" URL="../Modules/PI_M126_E816Piezo.llb/PI_Stages_Main.vi"/>
					<Item Name="SetSingleAxisPos.vi" Type="VI" URL="../Modules/PI_M126_E816Piezo.llb/SetSingleAxisPos.vi"/>
				</Item>
				<Item Name="PI Stage Control" Type="Folder">
					<Item Name="#7.vi" Type="VI" URL="../Modules/PI Stage Control.llb/#7.vi"/>
					<Item Name="*IDN?.vi" Type="VI" URL="../Modules/PI Stage Control.llb/*IDN?.vi"/>
					<Item Name="Assign values from string to axes.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Assign values from string to axes.vi"/>
					<Item Name="BDR.vi" Type="VI" URL="../Modules/PI Stage Control.llb/BDR.vi"/>
					<Item Name="Build command substring.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Build command substring.vi"/>
					<Item Name="Build query command substring.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Build query command substring.vi"/>
					<Item Name="Close connection if open.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Close connection if open.vi"/>
					<Item Name="Commanded axes connected?.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Commanded axes connected?.vi"/>
					<Item Name="Commanded stage name available?.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Commanded stage name available?.vi"/>
					<Item Name="Controller names.ctl" Type="VI" URL="../Modules/PI Stage Control.llb/Controller names.ctl"/>
					<Item Name="CST.vi" Type="VI" URL="../Modules/PI Stage Control.llb/CST.vi"/>
					<Item Name="Define connected axes.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Define connected axes.vi"/>
					<Item Name="ERR?.vi" Type="VI" URL="../Modules/PI Stage Control.llb/ERR?.vi"/>
					<Item Name="GCSTranslator DLL Functions.vi" Type="VI" URL="../Modules/PI Stage Control.llb/GCSTranslator DLL Functions.vi"/>
					<Item Name="Get lines from string.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Get lines from string.vi"/>
					<Item Name="Global1.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Global1.vi"/>
					<Item Name="Global2.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Global2.vi"/>
					<Item Name="INI.vi" Type="VI" URL="../Modules/PI Stage Control.llb/INI.vi"/>
					<Item Name="Longlasting one-axis command.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Longlasting one-axis command.vi"/>
					<Item Name="MOV?.vi" Type="VI" URL="../Modules/PI Stage Control.llb/MOV?.vi"/>
					<Item Name="MPL.vi" Type="VI" URL="../Modules/PI Stage Control.llb/MPL.vi"/>
					<Item Name="PI Open Interface.vi" Type="VI" URL="../Modules/PI Stage Control.llb/PI Open Interface.vi"/>
					<Item Name="PI Receive String.vi" Type="VI" URL="../Modules/PI Stage Control.llb/PI Receive String.vi"/>
					<Item Name="PI ReceiveNCharacters RS232.vi" Type="VI" URL="../Modules/PI Stage Control.llb/PI ReceiveNCharacters RS232.vi"/>
					<Item Name="PI ReceiveString GPIB.vi" Type="VI" URL="../Modules/PI Stage Control.llb/PI ReceiveString GPIB.vi"/>
					<Item Name="PI Send String.vi" Type="VI" URL="../Modules/PI Stage Control.llb/PI Send String.vi"/>
					<Item Name="POS?.vi" Type="VI" URL="../Modules/PI Stage Control.llb/POS?.vi"/>
					<Item Name="SAI?.vi" Type="VI" URL="../Modules/PI Stage Control.llb/SAI?.vi"/>
					<Item Name="Split num query command.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Split num query command.vi"/>
					<Item Name="SVA?.vi" Type="VI" URL="../Modules/PI Stage Control.llb/SVA?.vi"/>
					<Item Name="SVO.vi" Type="VI" URL="../Modules/PI Stage Control.llb/SVO.vi"/>
					<Item Name="VOL?.vi" Type="VI" URL="../Modules/PI Stage Control.llb/VOL?.vi"/>
					<Item Name="MOV.vi" Type="VI" URL="../Modules/PI Stage Control.llb/MOV.vi"/>
					<Item Name="VST?.vi" Type="VI" URL="../Modules/PI Stage Control.llb/VST?.vi"/>
					<Item Name="Wait for answer of longlasting command.vi" Type="VI" URL="../Modules/PI Stage Control.llb/Wait for answer of longlasting command.vi"/>
					<Item Name="C843_E665_Configuration_Self.vi" Type="VI" URL="../Modules/PI Stage Control.llb/C843_E665_Configuration_Self.vi"/>
				</Item>
			</Item>
		</Item>
		<Item Name="Setups" Type="Folder">
			<Item Name="Setup_D012_L.vi" Type="VI" URL="../Setups/Setup_D012_L.vi"/>
			<Item Name="Setup_D012_R.vi" Type="VI" URL="../Setups/Setup_D012_R.vi"/>
		</Item>
		<Item Name="GetBeadCornerPos.vi" Type="VI" URL="../BeadTracker2.llb/GetBeadCornerPos.vi"/>
		<Item Name="MainUI.vi" Type="VI" URL="../BeadTracker2.llb/MainUI.vi"/>
		<Item Name="SimpleCameraTest.vi" Type="VI" URL="../SimpleCameraTest.vi"/>
		<Item Name="TestGetQueues.vi" Type="VI" URL="../TestGetQueues.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Application Directory.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Application Directory.vi"/>
				<Item Name="BuildHelpPath.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/BuildHelpPath.vi"/>
				<Item Name="Bytes At Serial Port.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/serial.llb/Bytes At Serial Port.vi"/>
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
				<Item Name="IMAQ Start.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Start.vi"/>
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
				<Item Name="NI_PackedLibraryUtility.lvlib" Type="Library" URL="/&lt;vilib&gt;/Utility/LVLibp/NI_PackedLibraryUtility.lvlib"/>
				<Item Name="NI_Vision_Development_Module.lvlib" Type="Library" URL="/&lt;vilib&gt;/vision/NI_Vision_Development_Module.lvlib"/>
				<Item Name="Not Found Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Not Found Dialog.vi"/>
				<Item Name="Open File+.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Open File+.vi"/>
				<Item Name="Open Serial Driver.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_sersup.llb/Open Serial Driver.vi"/>
				<Item Name="Read File+ (string).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read File+ (string).vi"/>
				<Item Name="Read From Spreadsheet File (DBL).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read From Spreadsheet File (DBL).vi"/>
				<Item Name="Read From Spreadsheet File (I64).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read From Spreadsheet File (I64).vi"/>
				<Item Name="Read From Spreadsheet File (string).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read From Spreadsheet File (string).vi"/>
				<Item Name="Read From Spreadsheet File.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read From Spreadsheet File.vi"/>
				<Item Name="Read Lines From File.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Read Lines From File.vi"/>
				<Item Name="Search and Replace Pattern.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Search and Replace Pattern.vi"/>
				<Item Name="Serial Port Init.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/serial.llb/Serial Port Init.vi"/>
				<Item Name="Serial Port Read.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/serial.llb/Serial Port Read.vi"/>
				<Item Name="Serial Port Write.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/serial.llb/Serial Port Write.vi"/>
				<Item Name="serpConfig.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/serial.llb/serpConfig.vi"/>
				<Item Name="SessionLookUp.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/SessionLookUp.vi"/>
				<Item Name="Set Bold Text.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set Bold Text.vi"/>
				<Item Name="Set String Value.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set String Value.vi"/>
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
			<Item Name="AccurateTickCount.vi" Type="VI" URL="../AccurateTickCount.vi"/>
			<Item Name="GCSTranslator.dll" Type="Document" URL="../Modules/GCSTranslator.dll"/>
			<Item Name="imaq.dll" Type="Document" URL="imaq.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="kernel32.dll" Type="Document" URL="kernel32.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="lvanlys.dll" Type="Document" URL="/C/Program Files (x86)/National Instruments/LabVIEW 2011/resource/lvanlys.dll"/>
			<Item Name="nivision.dll" Type="Document" URL="nivision.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="QTrkCheckForDLL.vi" Type="VI" URL="../QTrk.llb/QTrkCheckForDLL.vi"/>
			<Item Name="QTrkCreate.vi" Type="VI" URL="../QTrk.llb/QTrkCreate.vi"/>
			<Item Name="QTrkDLL.vi" Type="VI" URL="../QTrk.llb/QTrkDLL.vi"/>
			<Item Name="QTrkFlush.vi" Type="VI" URL="../QTrk.llb/QTrkFlush.vi"/>
			<Item Name="QTrkFree.vi" Type="VI" URL="../QTrk.llb/QTrkFree.vi"/>
			<Item Name="QTrkFreeAllTrackers.vi" Type="VI" URL="../QTrk.llb/QTrkFreeAllTrackers.vi"/>
			<Item Name="QTrkGetResultCount.vi" Type="VI" URL="../QTrk.llb/QTrkGetResultCount.vi"/>
			<Item Name="QTrkGetZLUT.vi" Type="VI" URL="../QTrk.llb/QTrkGetZLUT.vi"/>
			<Item Name="QTrkInstance.ctl" Type="VI" URL="../QTrk.llb/QTrkInstance.ctl"/>
			<Item Name="QTrkIsIdle.vi" Type="VI" URL="../QTrk.llb/QTrkIsIdle.vi"/>
			<Item Name="QTrkPixelDataType.ctl" Type="VI" URL="../QTrk.llb/QTrkPixelDataType.ctl"/>
			<Item Name="QTrkQueueFrame.vi" Type="VI" URL="../QTrk.llb/QTrkQueueFrame.vi"/>
			<Item Name="QTrkSelectDLLDialog.vi" Type="VI" URL="../QTrk.llb/QTrkSelectDLLDialog.vi"/>
			<Item Name="QTrkSettings.ctl" Type="VI" URL="../QTrk.llb/QTrkSettings.ctl"/>
			<Item Name="QTrkSetZLUT.vi" Type="VI" URL="../QTrk.llb/QTrkSetZLUT.vi"/>
			<Item Name="SelectBeads.vi" Type="VI" URL="../BeadTracker2.llb/SelectBeads.vi"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
