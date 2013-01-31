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
		<Item Name="BT2_PluginConfig.vi" Type="VI" URL="../BT2_PluginConfig.vi"/>
		<Item Name="CameraInterface.lvclass" Type="LVClass" URL="../BeadTracker2.llb/CameraInterface.lvclass"/>
		<Item Name="CameraViewUI.vi" Type="VI" URL="../BeadTracker2.llb/CameraViewUI.vi"/>
		<Item Name="GlobalVar_Plugins.vi" Type="VI" URL="../BeadTracker2.llb/GlobalVar_Plugins.vi"/>
		<Item Name="MotorDriverInterface.lvclass" Type="LVClass" URL="../BeadTracker2.llb/MotorDriverInterface.lvclass"/>
		<Item Name="PluginConfig.ctl" Type="VI" URL="../BeadTracker2.llb/PluginConfig.ctl"/>
		<Item Name="SimpleCameraTest.vi" Type="VI" URL="../BeadTracker2.llb/SimpleCameraTest.vi"/>
		<Item Name="VisionExpressCamera.lvclass" Type="LVClass" URL="../BeadTracker2.llb/VisionExpressCamera.lvclass"/>
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
				<Item Name="Vision Acquisition CalculateFPS.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/Vision Acquisition Express Utility VIs.llb/Vision Acquisition CalculateFPS.vi"/>
				<Item Name="Vision Acquisition IMAQ Filter Stop Trigger Error.vi" Type="VI" URL="/&lt;vilib&gt;/vision/driver/Vision Acquisition Express Utility VIs.llb/Vision Acquisition IMAQ Filter Stop Trigger Error.vi"/>
			</Item>
			<Item Name="imaq.dll" Type="Document" URL="imaq.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
