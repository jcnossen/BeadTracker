%cudatcx;	cudatcy;	cudatcz;	cudax;	cuday;	cudaz;	cpux;	cpuy;	cpuz;	
d=dlmread('cmpresults.txt');

cpu = d(:,7:9);
gtc = d(:,1:3);
gm = d(:,4:6);
dtc = gtc-cpu;
dm = gm-cpu;
fprintf('GPU(tex): Mean offset X %f, Y: %f, Z: %f\n', mean(dtc));
fprintf('GPU(mem): Mean offset X %f, Y: %f, Z: %f\n', mean(dm));

fprintf('GPU(tex): Stdev offset X %f, Y: %f, Z: %f\n', std(dtc));
fprintf('GPU(mem): Stdev offset X %f, Y: %f, Z: %f\n', std(dm));
