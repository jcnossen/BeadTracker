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
		<Item Name="BeadTracker2.llb" Type="Folder" URL="../BeadTracker2.llb">
			<Property Name="NI.DISK" Type="Bool">true</Property>
		</Item>
		<Item Name="Modules" Type="Folder">
			<Property Name="NI.SortType" Type="Int">3</Property>
			<Item Name="Cameras" Type="Folder">
				<Item Name="VisionExpressCamera.llb" Type="Folder" URL="../Modules/VisionExpressCamera.llb">
					<Property Name="NI.DISK" Type="Bool">true</Property>
				</Item>
			</Item>
			<Item Name="MotorControl" Type="Folder">
				<Item Name="PI_M126_E816Piezo.llb" Type="Folder" URL="../Modules/PI_M126_E816Piezo.llb">
					<Property Name="NI.DISK" Type="Bool">true</Property>
				</Item>
			</Item>
		</Item>
		<Item Name="Setups" Type="Folder" URL="../Setups">
			<Property Name="NI.DISK" Type="Bool">true</Property>
		</Item>
		<Item Name="SimpleCameraTest.vi" Type="VI" URL="../SimpleCameraTest.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Image Type" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Image Type"/>
				<Item Name="IMAQ Attribute.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Attribute.vi"/>
				<Item Name="IMAQ Close.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqhl.llb/IMAQ Close.vi"/>
				<Item Name="IMAQ Configure Buffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Configure Buffer.vi"/>
				<Item Name="IMAQ Configure List.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Configure List.vi"/>
				<Item Name="IMAQ Create" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Create"/>
				<Item Name="IMAQ Extract Buffer Old Style.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/IMAQ Extract Buffer Old Style.vi"/>
				<Item Name="IMAQ Extract Buffer.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/imaqll.llb/IMAQ Extract Buffer.vi"/>
				<Item Name="IMAQ Image.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/IMAQ Image.ctl"/>
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
				<Item Name="SessionLookUp.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/DLLCalls.llb/SessionLookUp.vi"/>
				<Item Name="VISA Configure Serial Port" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port"/>
				<Item Name="VISA Configure Serial Port (Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Instr).vi"/>
				<Item Name="VISA Configure Serial Port (Serial Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Serial Instr).vi"/>
				<Item Name="Vision Acquisition CalculateFPS.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/Vision Acquisition Express Utility VIs.llb/Vision Acquisition CalculateFPS.vi"/>
				<Item Name="Vision Acquisition IMAQ Filter Stop Trigger Error.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/Vision Acquisition Express Utility VIs.llb/Vision Acquisition IMAQ Filter Stop Trigger Error.vi"/>
			</Item>
			<Item Name="#5.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Special command.llb/#5.vi"/>
			<Item Name="#5_old.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Old commands.llb/#5_old.vi"/>
			<Item Name="#7.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Special command.llb/#7.vi"/>
			<Item Name="#24.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Special command.llb/#24.vi"/>
			<Item Name="*IDN?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/*IDN?.vi"/>
			<Item Name="Analog FGlobal.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Analog control.llb/Analog FGlobal.vi"/>
			<Item Name="Analog Functions.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Analog control.llb/Analog Functions.vi"/>
			<Item Name="Analog Receive String.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Analog control.llb/Analog Receive String.vi"/>
			<Item Name="Assign booleans from string to axes.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Assign booleans from string to axes.vi"/>
			<Item Name="Assign NaN for chosen axes.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Assign NaN for chosen axes.vi"/>
			<Item Name="Assign values from string to axes.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Assign values from string to axes.vi"/>
			<Item Name="Available Analog Commands.ctl" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Analog control.llb/Available Analog Commands.ctl"/>
			<Item Name="Available DLL interfaces.ctl" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/Available DLL interfaces.ctl"/>
			<Item Name="Available DLLs.ctl" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/Available DLLs.ctl"/>
			<Item Name="Available interfaces.ctl" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/Available interfaces.ctl"/>
			<Item Name="BDR.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/BDR.vi"/>
			<Item Name="Build channel query command substring.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Build channel query command substring.vi"/>
			<Item Name="Build command substring.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Build command substring.vi"/>
			<Item Name="Build query command substring.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Build query command substring.vi"/>
			<Item Name="C843_Configuration_Setup.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/C843_Configuration_Setup.vi"/>
			<Item Name="CameraInterface.lvclass" Type="LVClass" URL="../BeadTracker2.llb/CameraInterface.lvclass"/>
			<Item Name="Close connection if open.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/Close connection if open.vi"/>
			<Item Name="CmdType_CameraIn.ctl" Type="VI" URL="/C/Documents and Settings/jcnossen1/Local Settings/Temp/tarc38916960_1.llb/CmdType_CameraIn.ctl"/>
			<Item Name="CmdType_CameraIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdType_CameraIn.ctl"/>
			<Item Name="CmdType_CameraOut.ctl" Type="VI" URL="/C/Documents and Settings/jcnossen1/Local Settings/Temp/tarc38916960_1.llb/CmdType_CameraOut.ctl"/>
			<Item Name="CmdType_CameraOut.ctl" Type="VI" URL="../BeadTracker2.llb/CmdType_CameraOut.ctl"/>
			<Item Name="CmdType_MotorIn.ctl" Type="VI" URL="/C/Documents and Settings/jcnossen1/Local Settings/Temp/tarc38916960_1.llb/CmdType_MotorIn.ctl"/>
			<Item Name="CmdType_MotorIn.ctl" Type="VI" URL="../BeadTracker2.llb/CmdType_MotorIn.ctl"/>
			<Item Name="Commanded axes connected?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Commanded axes connected?.vi"/>
			<Item Name="Commanded stage name available?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Commanded stage name available?.vi"/>
			<Item Name="Controller names.ctl" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/Controller names.ctl"/>
			<Item Name="CST.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Special command.llb/CST.vi"/>
			<Item Name="CST?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Special command.llb/CST?.vi"/>
			<Item Name="Cut out additional spaces.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Cut out additional spaces.vi"/>
			<Item Name="Define axes to command from boolean array.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Define axes to command from boolean array.vi"/>
			<Item Name="Define connected axes.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/Define connected axes.vi"/>
			<Item Name="Define connected stages with dialog.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Define connected stages with dialog.vi"/>
			<Item Name="Define connected systems (Array).vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/Define connected systems (Array).vi"/>
			<Item Name="DFH.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Limits.llb/DFH.vi"/>
			<Item Name="ERR?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/ERR?.vi"/>
			<Item Name="GCSTranslateError.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/GCSTranslateError.vi"/>
			<Item Name="GCSTranslator DLL Functions.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/GCSTranslator DLL Functions.vi"/>
			<Item Name="GCSTranslator.dll" Type="Document" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/GCSTranslator.dll"/>
			<Item Name="General wait for movement to stop.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/General wait for movement to stop.vi"/>
			<Item Name="Get all axes.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Get all axes.vi"/>
			<Item Name="Get arrays without blanks.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Get arrays without blanks.vi"/>
			<Item Name="Get lines from string.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Get lines from string.vi"/>
			<Item Name="Global Analog.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Analog control.llb/Global Analog.vi"/>
			<Item Name="Global DaisyChain.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/Global DaisyChain.vi"/>
			<Item Name="Global1.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/Global1.vi"/>
			<Item Name="Global2 (Array).vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/Global2 (Array).vi"/>
			<Item Name="GlobalVariables.vi" Type="VI" URL="/C/Documents and Settings/jcnossen1/Local Settings/Temp/tarc38916960_1.llb/GlobalVariables.vi"/>
			<Item Name="GlobalVariables.vi" Type="VI" URL="../BeadTracker2.llb/GlobalVariables.vi"/>
			<Item Name="imaq.dll" Type="Document" URL="imaq.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="INI.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Special command.llb/INI.vi"/>
			<Item Name="Initialize Global1.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/Initialize Global1.vi"/>
			<Item Name="Initialize Global2.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/Initialize Global2.vi"/>
			<Item Name="Is DaisyChain open.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/Is DaisyChain open.vi"/>
			<Item Name="LIM?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Limits.llb/LIM?.vi"/>
			<Item Name="Longlasting one-axis command.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Longlasting one-axis command.vi"/>
			<Item Name="MainUI.vi" Type="VI" URL="../BeadTracker2.llb/MainUI.vi"/>
			<Item Name="MNL.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Limits.llb/MNL.vi"/>
			<Item Name="MotorDriverInterface.lvclass" Type="LVClass" URL="../BeadTracker2.llb/MotorDriverInterface.lvclass"/>
			<Item Name="MOV.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/MOV.vi"/>
			<Item Name="MOV?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/MOV?.vi"/>
			<Item Name="MPL.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Limits.llb/MPL.vi"/>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="ONT?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/ONT?.vi"/>
			<Item Name="PI Open Interface of one system.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/PI Open Interface of one system.vi"/>
			<Item Name="PI Receive String.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/PI Receive String.vi"/>
			<Item Name="PI Send String.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/PI Send String.vi"/>
			<Item Name="PI VISA Receive Characters.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/PI VISA Receive Characters.vi"/>
			<Item Name="PluginConfig.ctl" Type="VI" URL="../BeadTracker2.llb/PluginConfig.ctl"/>
			<Item Name="POS?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/POS?.vi"/>
			<Item Name="QueueListType.ctl" Type="VI" URL="/C/Documents and Settings/jcnossen1/Local Settings/Temp/tarc38916960_1.llb/QueueListType.ctl"/>
			<Item Name="QueueListType.ctl" Type="VI" URL="../BeadTracker2.llb/QueueListType.ctl"/>
			<Item Name="REF.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Limits.llb/REF.vi"/>
			<Item Name="REF?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Limits.llb/REF?.vi"/>
			<Item Name="Return space.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Return space.vi"/>
			<Item Name="RON.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Limits.llb/RON.vi"/>
			<Item Name="RON?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Limits.llb/RON?.vi"/>
			<Item Name="SAI?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/SAI?.vi"/>
			<Item Name="Select values for chosen axes.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Select values for chosen axes.vi"/>
			<Item Name="Set RON and return RON status.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Set RON and return RON status.vi"/>
			<Item Name="Split num query command.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Old commands.llb/Split num query command.vi"/>
			<Item Name="STA?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Special command.llb/STA?.vi"/>
			<Item Name="String with ASCII code conversion.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/String with ASCII code conversion.vi"/>
			<Item Name="Substract axes array subset from axes array.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Substract axes array subset from axes array.vi"/>
			<Item Name="SVA?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/PZT voltage.llb/SVA?.vi"/>
			<Item Name="SVO.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/General command.llb/SVO.vi"/>
			<Item Name="Termination character.ctl" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Communication.llb/Termination character.ctl"/>
			<Item Name="TMN?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Limits.llb/TMN?.vi"/>
			<Item Name="TMX?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Limits.llb/TMX?.vi"/>
			<Item Name="TPC?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Special command.llb/TPC?.vi"/>
			<Item Name="TSC?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Optical or Analog Input.llb/TSC?.vi"/>
			<Item Name="VisionExpressCamera.lvclass" Type="LVClass" URL="../Modules/VisionExpressCamera.llb/VisionExpressCamera.lvclass"/>
			<Item Name="VOL?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/PZT voltage.llb/VOL?.vi"/>
			<Item Name="VST?.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Special command.llb/VST?.vi"/>
			<Item Name="Wait for answer of longlasting command.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Wait for answer of longlasting command.vi"/>
			<Item Name="Wait for axes to stop.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Support.llb/Wait for axes to stop.vi"/>
			<Item Name="Wait for hexapod system axes to stop.vi" Type="VI" URL="/C/Program Files/PI/Merged_GCS_LabVIEW/Low Level/Old commands.llb/Wait for hexapod system axes to stop.vi"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
