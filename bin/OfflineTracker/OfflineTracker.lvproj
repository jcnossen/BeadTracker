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
		<Item Name="JTrkAPI.llb" Type="Folder" URL="../JTrkAPI.llb">
			<Property Name="NI.DISK" Type="Bool">true</Property>
		</Item>
		<Item Name="OfflineTrackerJTrk" Type="Folder">
			<Item Name="boxoverlayimage.vi" Type="VI" URL="../OfflineTrackerJTrk/boxoverlayimage.vi"/>
			<Item Name="Check Limits.vi" Type="VI" URL="../OfflineTrackerJTrk/Check Limits.vi"/>
			<Item Name="CheckROIEdges (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/CheckROIEdges (SubVI).vi"/>
			<Item Name="CleanIT (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/CleanIT (SubVI).vi"/>
			<Item Name="CleanROIs (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/CleanROIs (SubVI).vi"/>
			<Item Name="CompareZProfile2.vi" Type="VI" URL="../OfflineTrackerJTrk/CompareZProfile2.vi"/>
			<Item Name="CP que.ctl" Type="VI" URL="../OfflineTrackerJTrk/CP que.ctl"/>
			<Item Name="datahandleMVL.vi" Type="VI" URL="../OfflineTrackerJTrk/datahandleMVL.vi"/>
			<Item Name="draw rectangles.vi" Type="VI" URL="../OfflineTrackerJTrk/draw rectangles.vi"/>
			<Item Name="General Polynomial Fit with SD.vi" Type="VI" URL="../OfflineTrackerJTrk/General Polynomial Fit with SD.vi"/>
			<Item Name="Get Max 1D 3p.vi" Type="VI" URL="../OfflineTrackerJTrk/Get Max 1D 3p.vi"/>
			<Item Name="Global PhaseStandalone.vi" Type="VI" URL="../OfflineTrackerJTrk/Global PhaseStandalone.vi"/>
			<Item Name="imagereque.vi" Type="VI" URL="../OfflineTrackerJTrk/imagereque.vi"/>
			<Item Name="load images.vi" Type="VI" URL="../OfflineTrackerJTrk/load images.vi"/>
			<Item Name="LoadOneImageviaPath.vi" Type="VI" URL="../OfflineTrackerJTrk/LoadOneImageviaPath.vi"/>
			<Item Name="LoadZlut.vi" Type="VI" URL="../OfflineTrackerJTrk/LoadZlut.vi"/>
			<Item Name="MainProgram.vi" Type="VI" URL="../OfflineTrackerJTrk/MainProgram.vi"/>
			<Item Name="MakeAProfileJK.vi" Type="VI" URL="../OfflineTrackerJTrk/MakeAProfileJK.vi"/>
			<Item Name="MakeBigTemplate (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/MakeBigTemplate (SubVI).vi"/>
			<Item Name="MakeCustomWindow (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/MakeCustomWindow (SubVI).vi"/>
			<Item Name="MakePhiAxis (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/MakePhiAxis (SubVI).vi"/>
			<Item Name="MakeRadialAxis (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/MakeRadialAxis (SubVI).vi"/>
			<Item Name="MakeZlut.vi" Type="VI" URL="../OfflineTrackerJTrk/MakeZlut.vi"/>
			<Item Name="MinusMean2D (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/MinusMean2D (SubVI).vi"/>
			<Item Name="OfflineTrackerHandleMVL2.vi" Type="VI" URL="../OfflineTrackerJTrk/OfflineTrackerHandleMVL2.vi"/>
			<Item Name="OneQuadProfile (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/OneQuadProfile (SubVI).vi"/>
			<Item Name="parabola fit2.vi" Type="VI" URL="../OfflineTrackerJTrk/parabola fit2.vi"/>
			<Item Name="parabola fitZ.vi" Type="VI" URL="../OfflineTrackerJTrk/parabola fitZ.vi"/>
			<Item Name="pix2um&amp;refsubstract.vi" Type="VI" URL="../OfflineTrackerJTrk/pix2um&amp;refsubstract.vi"/>
			<Item Name="Plot_em.vi" Type="VI" URL="../OfflineTrackerJTrk/Plot_em.vi"/>
			<Item Name="QuadCombine (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/QuadCombine (SubVI).vi"/>
			<Item Name="QuadExtraPass (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/QuadExtraPass (SubVI).vi"/>
			<Item Name="RECenterROI (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/RECenterROI (SubVI).vi"/>
			<Item Name="RemovenearestROI.vi" Type="VI" URL="../OfflineTrackerJTrk/RemovenearestROI.vi"/>
			<Item Name="roi2xy.vi" Type="VI" URL="../OfflineTrackerJTrk/roi2xy.vi"/>
			<Item Name="ROIAutoSearch.vi" Type="VI" URL="../OfflineTrackerJTrk/ROIAutoSearch.vi"/>
			<Item Name="ROICenter2LTRB.vi" Type="VI" URL="../OfflineTrackerJTrk/ROICenter2LTRB.vi"/>
			<Item Name="ROIlistautofill.vi" Type="VI" URL="../OfflineTrackerJTrk/ROIlistautofill.vi"/>
			<Item Name="SaveData.vi" Type="VI" URL="../OfflineTrackerJTrk/SaveData.vi"/>
			<Item Name="Select Bests (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/Select Bests (SubVI).vi"/>
			<Item Name="SetBeads.vi" Type="VI" URL="../OfflineTrackerJTrk/SetBeads.vi"/>
			<Item Name="ShowBead.vi" Type="VI" URL="../OfflineTrackerJTrk/ShowBead.vi"/>
			<Item Name="SortOnKey (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/SortOnKey (SubVI).vi"/>
			<Item Name="Stack1LUT(SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/Stack1LUT(SubVI).vi"/>
			<Item Name="Swapit2D (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/Swapit2D (SubVI).vi"/>
			<Item Name="Tracker.vi" Type="VI" URL="../OfflineTrackerJTrk/Tracker.vi"/>
			<Item Name="trackerhandle.vi" Type="VI" URL="../OfflineTrackerJTrk/trackerhandle.vi"/>
			<Item Name="Trackit.vi" Type="VI" URL="../OfflineTrackerJTrk/Trackit.vi"/>
			<Item Name="Xcor1D (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/Xcor1D (SubVI).vi"/>
			<Item Name="Xcorimages (SubVI).vi" Type="VI" URL="../OfflineTrackerJTrk/Xcorimages (SubVI).vi"/>
			<Item Name="xy2roi.vi" Type="VI" URL="../OfflineTrackerJTrk/xy2roi.vi"/>
			<Item Name="XY_GetCenterOfMass.vi" Type="VI" URL="../OfflineTrackerJTrk/XY_GetCenterOfMass.vi"/>
		</Item>
		<Item Name="jtrk-labview.dll" Type="Document" URL="../jtrk-labview.dll"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Image Type" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Image Type"/>
				<Item Name="Image Unit" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Image Unit"/>
				<Item Name="IMAQ GetImageInfo" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ GetImageInfo"/>
				<Item Name="IMAQ Image Cluster to Image Datatype.vi" Type="VI" URL="/&lt;vilib&gt;/vision/DatatypeConversion.llb/IMAQ Image Cluster to Image Datatype.vi"/>
				<Item Name="IMAQ Image.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/IMAQ Image.ctl"/>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
		</Item>
		<Item Name="Build Specifications" Type="Build">
			<Item Name="My Application" Type="EXE">
				<Property Name="App_copyErrors" Type="Bool">true</Property>
				<Property Name="App_INI_aliasGUID" Type="Str">{1E82AA50-88F0-4B5A-A441-4FDD94C0BED3}</Property>
				<Property Name="App_INI_GUID" Type="Str">{05E84782-3715-49A1-A3F6-97CD37B55E91}</Property>
				<Property Name="Bld_buildCacheID" Type="Str">{CB757018-9BF0-4448-B727-EB7C956EDD37}</Property>
				<Property Name="Bld_buildSpecName" Type="Str">My Application</Property>
				<Property Name="Bld_excludeLibraryItems" Type="Bool">true</Property>
				<Property Name="Bld_excludePolymorphicVIs" Type="Bool">true</Property>
				<Property Name="Bld_localDestDir" Type="Path">../builds/NI_AB_PROJECTNAME/My Application</Property>
				<Property Name="Bld_localDestDirType" Type="Str">relativeToCommon</Property>
				<Property Name="Bld_modifyLibraryFile" Type="Bool">true</Property>
				<Property Name="Bld_previewCacheID" Type="Str">{C7BB0533-C39D-493C-836C-3780F46CCD4C}</Property>
				<Property Name="Destination[0].destName" Type="Str">Application.exe</Property>
				<Property Name="Destination[0].path" Type="Path">../builds/NI_AB_PROJECTNAME/My Application/Application.exe</Property>
				<Property Name="Destination[0].preserveHierarchy" Type="Bool">true</Property>
				<Property Name="Destination[0].type" Type="Str">App</Property>
				<Property Name="Destination[1].destName" Type="Str">Support Directory</Property>
				<Property Name="Destination[1].path" Type="Path">../builds/NI_AB_PROJECTNAME/My Application/data</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="Source[0].itemID" Type="Str">{A5E1E16F-9CB5-4B28-ACB0-C13FC924E94B}</Property>
				<Property Name="Source[0].type" Type="Str">Container</Property>
				<Property Name="Source[1].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[1].itemID" Type="Ref"></Property>
				<Property Name="Source[1].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[1].type" Type="Str">VI</Property>
				<Property Name="Source[2].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[2].itemID" Type="Ref">/My Computer/jtrk-labview.dll</Property>
				<Property Name="Source[2].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[3].Container.applyInclusion" Type="Bool">true</Property>
				<Property Name="Source[3].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[3].itemID" Type="Ref"></Property>
				<Property Name="Source[3].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[3].type" Type="Str">Container</Property>
				<Property Name="SourceCount" Type="Int">4</Property>
				<Property Name="TgtF_companyName" Type="Str">Delft University of Technology</Property>
				<Property Name="TgtF_fileDescription" Type="Str">My Application</Property>
				<Property Name="TgtF_fileVersion.major" Type="Int">1</Property>
				<Property Name="TgtF_internalName" Type="Str">My Application</Property>
				<Property Name="TgtF_legalCopyright" Type="Str">Copyright © 2012 Delft University of Technology</Property>
				<Property Name="TgtF_productName" Type="Str">My Application</Property>
				<Property Name="TgtF_targetfileGUID" Type="Str">{E8686247-7BC5-4098-A8AC-AF69DACCF324}</Property>
				<Property Name="TgtF_targetfileName" Type="Str">Application.exe</Property>
			</Item>
			<Item Name="OfflineJTrk" Type="Source Distribution">
				<Property Name="Bld_buildCacheID" Type="Str">{9CF96E93-3713-4E88-BB86-01E5760D47B9}</Property>
				<Property Name="Bld_buildSpecName" Type="Str">OfflineJTrk</Property>
				<Property Name="Bld_excludedDirectory[0]" Type="Path">vi.lib</Property>
				<Property Name="Bld_excludedDirectory[0].pathType" Type="Str">relativeToAppDir</Property>
				<Property Name="Bld_excludedDirectory[1]" Type="Path">resource/objmgr</Property>
				<Property Name="Bld_excludedDirectory[1].pathType" Type="Str">relativeToAppDir</Property>
				<Property Name="Bld_excludedDirectory[2]" Type="Path">/H/My Documents/LabVIEW Data/InstCache</Property>
				<Property Name="Bld_excludedDirectory[3]" Type="Path">instr.lib</Property>
				<Property Name="Bld_excludedDirectory[3].pathType" Type="Str">relativeToAppDir</Property>
				<Property Name="Bld_excludedDirectory[4]" Type="Path">user.lib</Property>
				<Property Name="Bld_excludedDirectory[4].pathType" Type="Str">relativeToAppDir</Property>
				<Property Name="Bld_excludedDirectoryCount" Type="Int">5</Property>
				<Property Name="Bld_localDestDir" Type="Path">../builds/NI_AB_PROJECTNAME/OfflineTrk.llb</Property>
				<Property Name="Bld_localDestDirType" Type="Str">relativeToCommon</Property>
				<Property Name="Bld_previewCacheID" Type="Str">{A1169DA2-EEA4-430F-967D-5DBAAAAF25F2}</Property>
				<Property Name="Destination[0].destName" Type="Str">Destination Directory</Property>
				<Property Name="Destination[0].path" Type="Path">../builds/NI_AB_PROJECTNAME/OfflineTrk.llb</Property>
				<Property Name="Destination[0].type" Type="Str">LLB</Property>
				<Property Name="Destination[1].destName" Type="Str">Support Directory</Property>
				<Property Name="Destination[1].path" Type="Path">../builds/NI_AB_PROJECTNAME</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="Source[0].itemID" Type="Str">{6CD57291-1C30-4E1A-8E76-66BFCDB7E7B1}</Property>
				<Property Name="Source[0].type" Type="Str">Container</Property>
				<Property Name="Source[1].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[1].itemID" Type="Ref">/My Computer/jtrk-labview.dll</Property>
				<Property Name="Source[1].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[10].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[10].itemID" Type="Ref"></Property>
				<Property Name="Source[10].type" Type="Str">VI</Property>
				<Property Name="Source[11].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[11].itemID" Type="Ref"></Property>
				<Property Name="Source[11].type" Type="Str">VI</Property>
				<Property Name="Source[12].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[12].itemID" Type="Ref"></Property>
				<Property Name="Source[12].type" Type="Str">VI</Property>
				<Property Name="Source[13].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[13].itemID" Type="Ref"></Property>
				<Property Name="Source[13].type" Type="Str">VI</Property>
				<Property Name="Source[14].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[14].itemID" Type="Ref"></Property>
				<Property Name="Source[14].type" Type="Str">VI</Property>
				<Property Name="Source[15].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[15].itemID" Type="Ref"></Property>
				<Property Name="Source[15].type" Type="Str">VI</Property>
				<Property Name="Source[16].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[16].itemID" Type="Ref"></Property>
				<Property Name="Source[16].type" Type="Str">VI</Property>
				<Property Name="Source[17].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[17].itemID" Type="Ref"></Property>
				<Property Name="Source[17].type" Type="Str">VI</Property>
				<Property Name="Source[18].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[18].itemID" Type="Ref"></Property>
				<Property Name="Source[18].type" Type="Str">VI</Property>
				<Property Name="Source[19].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[19].itemID" Type="Ref"></Property>
				<Property Name="Source[19].type" Type="Str">VI</Property>
				<Property Name="Source[2].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[2].itemID" Type="Ref"></Property>
				<Property Name="Source[2].Library.allowMissingMembers" Type="Bool">true</Property>
				<Property Name="Source[2].type" Type="Str">Library</Property>
				<Property Name="Source[20].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[20].itemID" Type="Ref"></Property>
				<Property Name="Source[20].type" Type="Str">VI</Property>
				<Property Name="Source[21].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[21].itemID" Type="Ref"></Property>
				<Property Name="Source[21].type" Type="Str">VI</Property>
				<Property Name="Source[22].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[22].itemID" Type="Ref"></Property>
				<Property Name="Source[22].type" Type="Str">VI</Property>
				<Property Name="Source[23].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[23].itemID" Type="Ref"></Property>
				<Property Name="Source[23].type" Type="Str">VI</Property>
				<Property Name="Source[24].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[24].itemID" Type="Ref"></Property>
				<Property Name="Source[24].type" Type="Str">VI</Property>
				<Property Name="Source[25].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[25].itemID" Type="Ref"></Property>
				<Property Name="Source[25].type" Type="Str">VI</Property>
				<Property Name="Source[26].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[26].itemID" Type="Ref"></Property>
				<Property Name="Source[26].type" Type="Str">VI</Property>
				<Property Name="Source[27].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[27].itemID" Type="Ref"></Property>
				<Property Name="Source[27].type" Type="Str">VI</Property>
				<Property Name="Source[28].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[28].itemID" Type="Ref"></Property>
				<Property Name="Source[28].type" Type="Str">VI</Property>
				<Property Name="Source[29].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[29].itemID" Type="Ref"></Property>
				<Property Name="Source[29].type" Type="Str">VI</Property>
				<Property Name="Source[3].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[3].itemID" Type="Ref"></Property>
				<Property Name="Source[3].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[3].type" Type="Str">VI</Property>
				<Property Name="Source[3].VI.LLBtopLevel" Type="Bool">true</Property>
				<Property Name="Source[30].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[30].itemID" Type="Ref"></Property>
				<Property Name="Source[30].type" Type="Str">VI</Property>
				<Property Name="Source[31].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[31].itemID" Type="Ref"></Property>
				<Property Name="Source[31].type" Type="Str">VI</Property>
				<Property Name="Source[32].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[32].itemID" Type="Ref"></Property>
				<Property Name="Source[32].type" Type="Str">VI</Property>
				<Property Name="Source[33].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[33].itemID" Type="Ref"></Property>
				<Property Name="Source[33].type" Type="Str">VI</Property>
				<Property Name="Source[34].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[34].itemID" Type="Ref"></Property>
				<Property Name="Source[34].type" Type="Str">VI</Property>
				<Property Name="Source[35].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[35].itemID" Type="Ref"></Property>
				<Property Name="Source[35].type" Type="Str">VI</Property>
				<Property Name="Source[36].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[36].itemID" Type="Ref"></Property>
				<Property Name="Source[36].type" Type="Str">VI</Property>
				<Property Name="Source[37].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[37].itemID" Type="Ref"></Property>
				<Property Name="Source[37].type" Type="Str">VI</Property>
				<Property Name="Source[38].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[38].itemID" Type="Ref"></Property>
				<Property Name="Source[38].type" Type="Str">VI</Property>
				<Property Name="Source[39].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[39].itemID" Type="Ref"></Property>
				<Property Name="Source[39].type" Type="Str">VI</Property>
				<Property Name="Source[4].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[4].itemID" Type="Ref"></Property>
				<Property Name="Source[4].type" Type="Str">VI</Property>
				<Property Name="Source[40].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[40].itemID" Type="Ref"></Property>
				<Property Name="Source[40].type" Type="Str">VI</Property>
				<Property Name="Source[41].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[41].itemID" Type="Ref"></Property>
				<Property Name="Source[41].type" Type="Str">VI</Property>
				<Property Name="Source[42].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[42].itemID" Type="Ref"></Property>
				<Property Name="Source[42].type" Type="Str">VI</Property>
				<Property Name="Source[43].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[43].itemID" Type="Ref"></Property>
				<Property Name="Source[43].type" Type="Str">VI</Property>
				<Property Name="Source[44].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[44].itemID" Type="Ref"></Property>
				<Property Name="Source[44].type" Type="Str">VI</Property>
				<Property Name="Source[45].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[45].itemID" Type="Ref"></Property>
				<Property Name="Source[45].type" Type="Str">VI</Property>
				<Property Name="Source[46].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[46].itemID" Type="Ref"></Property>
				<Property Name="Source[46].type" Type="Str">VI</Property>
				<Property Name="Source[47].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[47].itemID" Type="Ref"></Property>
				<Property Name="Source[47].type" Type="Str">VI</Property>
				<Property Name="Source[48].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[48].itemID" Type="Ref"></Property>
				<Property Name="Source[48].type" Type="Str">VI</Property>
				<Property Name="Source[49].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[49].itemID" Type="Ref"></Property>
				<Property Name="Source[49].type" Type="Str">VI</Property>
				<Property Name="Source[5].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[5].itemID" Type="Ref"></Property>
				<Property Name="Source[5].type" Type="Str">VI</Property>
				<Property Name="Source[50].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[50].itemID" Type="Ref"></Property>
				<Property Name="Source[50].type" Type="Str">VI</Property>
				<Property Name="Source[51].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[51].itemID" Type="Ref"></Property>
				<Property Name="Source[51].type" Type="Str">VI</Property>
				<Property Name="Source[52].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[52].itemID" Type="Ref"></Property>
				<Property Name="Source[52].type" Type="Str">VI</Property>
				<Property Name="Source[53].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[53].itemID" Type="Ref"></Property>
				<Property Name="Source[53].type" Type="Str">VI</Property>
				<Property Name="Source[54].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[54].itemID" Type="Ref"></Property>
				<Property Name="Source[54].type" Type="Str">VI</Property>
				<Property Name="Source[55].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[55].itemID" Type="Ref"></Property>
				<Property Name="Source[55].type" Type="Str">VI</Property>
				<Property Name="Source[56].Container.applyInclusion" Type="Bool">true</Property>
				<Property Name="Source[56].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[56].itemID" Type="Ref">/My Computer/JTrkAPI.llb</Property>
				<Property Name="Source[56].sourceInclusion" Type="Str">Exclude</Property>
				<Property Name="Source[56].type" Type="Str">Container</Property>
				<Property Name="Source[6].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[6].itemID" Type="Ref"></Property>
				<Property Name="Source[6].type" Type="Str">VI</Property>
				<Property Name="Source[7].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[7].itemID" Type="Ref"></Property>
				<Property Name="Source[7].type" Type="Str">VI</Property>
				<Property Name="Source[8].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[8].itemID" Type="Ref"></Property>
				<Property Name="Source[8].type" Type="Str">VI</Property>
				<Property Name="Source[9].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[9].itemID" Type="Ref"></Property>
				<Property Name="Source[9].type" Type="Str">VI</Property>
				<Property Name="SourceCount" Type="Int">57</Property>
			</Item>
		</Item>
	</Item>
</Project>
