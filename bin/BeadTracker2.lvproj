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
		<Item Name="BeadTracker2.llb" Type="Folder">
			<Item Name="BuildZLUT.vi" Type="VI" URL="../BeadTracker2.llb/BuildZLUT.vi"/>
			<Item Name="CmdData_NewFrame.ctl" Type="VI" URL="../BeadTracker2.llb/CmdData_NewFrame.ctl"/>
			<Item Name="CmdData_SetMotorPos.ctl" Type="VI" URL="../BeadTracker2.llb/CmdData_SetMotorPos.ctl"/>
			<Item Name="CmdEnum_MotorIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdEnum_MotorIn.ctl"/>
			<Item Name="CmdType_CameraIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdType_CameraIn.ctl"/>
			<Item Name="CmdType_CameraOut.ctl" Type="VI" URL="../BeadTracker2.llb/CmdType_CameraOut.ctl"/>
			<Item Name="CmdType_MotorIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdType_MotorIn.ctl"/>
			<Item Name="CreateQueues.vi" Type="VI" URL="../BeadTracker2.llb/CreateQueues.vi"/>
			<Item Name="GlobalVariables.vi" Type="VI" URL="../BeadTracker2.llb/GlobalVariables.vi"/>
			<Item Name="MainUI.vi" Type="VI" URL="../BeadTracker2.llb/MainUI.vi"/>
			<Item Name="MeasureConfigType.ctl" Type="VI" URL="../BeadTracker2.llb/MeasureConfigType.ctl"/>
			<Item Name="MotorStateType.ctl" Type="VI" URL="../BeadTracker2.llb/MotorStateType.ctl"/>
			<Item Name="MotorUI.vi" Type="VI" URL="../BeadTracker2.llb/MotorUI.vi"/>
			<Item Name="QueueListType.ctl" Type="VI" URL="../BeadTracker2.llb/QueueListType.ctl"/>
			<Item Name="SendCameraCmd.vi" Type="VI" URL="../BeadTracker2.llb/SendCameraCmd.vi"/>
			<Item Name="SendMotorCmd.vi" Type="VI" URL="../BeadTracker2.llb/SendMotorCmd.vi"/>
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
					<Item Name="MoveMotorAxis.vi" Type="VI" URL="../BeadTracker2.llb/MoveMotorAxis.vi"/>
					<Item Name="MoveToPosition.vi" Type="VI" URL="../Modules/PI_M126_E816Piezo.llb/MoveToPosition.vi"/>
					<Item Name="PI_Stages_Main.vi" Type="VI" URL="../Modules/PI_M126_E816Piezo.llb/PI_Stages_Main.vi"/>
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
		<Item Name="SimpleCameraTest.vi" Type="VI" URL="../SimpleCameraTest.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Bytes At Serial Port.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/serial.llb/Bytes At Serial Port.vi"/>
				<Item Name="Image Type" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Image Type"/>
				<Item Name="IMAQ ArrayToImage" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ArrayToImage"/>
				<Item Name="IMAQ Attribute.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Attribute.vi"/>
				<Item Name="IMAQ Close.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Close.vi"/>
				<Item Name="IMAQ Configure Buffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Configure Buffer.vi"/>
				<Item Name="IMAQ Configure List.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Configure List.vi"/>
				<Item Name="IMAQ Create" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Create"/>
				<Item Name="IMAQ Extract Buffer Old Style.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/IMAQ Extract Buffer Old Style.vi"/>
				<Item Name="IMAQ Extract Buffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Extract Buffer.vi"/>
				<Item Name="IMAQ GetImageSize" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ GetImageSize"/>
				<Item Name="IMAQ Image.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/IMAQ Image.ctl"/>
				<Item Name="IMAQ ImageToArray" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ImageToArray"/>
				<Item Name="IMAQ Init.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Init.vi"/>
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
				<Item Name="LVPointDoubleTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVPointDoubleTypeDef.ctl"/>
				<Item Name="NI_AALBase.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALBase.lvlib"/>
				<Item Name="NI_AALPro.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALPro.lvlib"/>
				<Item Name="NI_Vision_Development_Module.lvlib" Type="Library" URL="/&lt;vilib&gt;/vision/NI_Vision_Development_Module.lvlib"/>
				<Item Name="Open Serial Driver.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_sersup.llb/Open Serial Driver.vi"/>
				<Item Name="Serial Port Init.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/serial.llb/Serial Port Init.vi"/>
				<Item Name="Serial Port Read.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/serial.llb/Serial Port Read.vi"/>
				<Item Name="Serial Port Write.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/serial.llb/Serial Port Write.vi"/>
				<Item Name="serpConfig.vi" Type="VI" URL="/&lt;vilib&gt;/Instr/serial.llb/serpConfig.vi"/>
				<Item Name="SessionLookUp.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/SessionLookUp.vi"/>
				<Item Name="VISA Configure Serial Port" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port"/>
				<Item Name="VISA Configure Serial Port (Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Instr).vi"/>
				<Item Name="VISA Configure Serial Port (Serial Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Serial Instr).vi"/>
				<Item Name="Vision Acquisition CalculateFPS.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/Vision Acquisition Express Utility VIs.llb/Vision Acquisition CalculateFPS.vi"/>
				<Item Name="Vision Acquisition IMAQ Filter Stop Trigger Error.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/Vision Acquisition Express Utility VIs.llb/Vision Acquisition IMAQ Filter Stop Trigger Error.vi"/>
			</Item>
			<Item Name="AccurateTickCount.vi" Type="VI" URL="../AccurateTickCount.vi"/>
			<Item Name="CleanIT (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/CleanIT (SubVI).vi"/>
			<Item Name="CleanROIs (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/CleanROIs (SubVI).vi"/>
			<Item Name="draw rectangles.vi" Type="VI" URL="../AutoBeadFinder.llb/draw rectangles.vi"/>
			<Item Name="GCSTranslator.dll" Type="Document" URL="../Modules/GCSTranslator.dll"/>
			<Item Name="imaq.dll" Type="Document" URL="imaq.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="kernel32.dll" Type="Document" URL="kernel32.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="lvanlys.dll" Type="Document" URL="/C/Program Files (x86)/National Instruments/LabVIEW 2011/resource/lvanlys.dll"/>
			<Item Name="MakeBigTemplate (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/MakeBigTemplate (SubVI).vi"/>
			<Item Name="MinusMean2D (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/MinusMean2D (SubVI).vi"/>
			<Item Name="nivision.dll" Type="Document" URL="nivision.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="QTrkSettings.ctl" Type="VI" URL="../QTrk.llb/QTrkSettings.ctl"/>
			<Item Name="RECenterROI (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/RECenterROI (SubVI).vi"/>
			<Item Name="RemovenearestROI.vi" Type="VI" URL="../AutoBeadFinder.llb/RemovenearestROI.vi"/>
			<Item Name="roi2xy.vi" Type="VI" URL="../AutoBeadFinder.llb/roi2xy.vi"/>
			<Item Name="ROIAutoSearch.vi" Type="VI" URL="../AutoBeadFinder.llb/ROIAutoSearch.vi"/>
			<Item Name="ROICenter2LTRB.vi" Type="VI" URL="../AutoBeadFinder.llb/ROICenter2LTRB.vi"/>
			<Item Name="ROIlistautofill.vi" Type="VI" URL="../AutoBeadFinder.llb/ROIlistautofill.vi"/>
			<Item Name="Select Bests (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/Select Bests (SubVI).vi"/>
			<Item Name="SelectBeads.vi" Type="VI" URL="../BeadTracker2.llb/SelectBeads.vi"/>
			<Item Name="SortOnKey (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/SortOnKey (SubVI).vi"/>
			<Item Name="Swapit2D (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/Swapit2D (SubVI).vi"/>
			<Item Name="Xcorimages (SubVI).vi" Type="VI" URL="../AutoBeadFinder.llb/Xcorimages (SubVI).vi"/>
			<Item Name="xy2roi.vi" Type="VI" URL="../AutoBeadFinder.llb/xy2roi.vi"/>
			<Item Name="XY_GetCenterOfMass.vi" Type="VI" URL="../AutoBeadFinder.llb/XY_GetCenterOfMass.vi"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
