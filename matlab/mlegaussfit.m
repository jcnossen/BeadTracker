% MATLAB implementation test
function function_wrapper = mlegaussfit()

    wrapper = {};
    wrapper.test = @test;
end


function [ Pestimate ] = fitgauss(smpimg, P, iterations)

    width =  size(smpimg,2);
    height =size(smpimg,1);
    
    mean_img = mean(smpimg(:));
	I0 = mean_img*0.5*numel(smpimg);
	bg = mean_img*0.5;
    
    sigma = P(3);

	r1oSq2Sigma = 1.0 / (sqrt(2) * sigma);
	r1oSq2PiSigma = 1.0 / (sqrt(2*pi) * sigma);
	r1oSq2PiSigma3 = 1.0 / (sqrt(2*pi) * sigma^2);

    posx = P(1);
    posy = P(2);
    
    [X,Y] = meshgrid(0:width-1,0:height-1);

    for i=1:iterations
        Xexp0 = (X-posx + .5) * r1oSq2Sigma;
        Yexp0 = (Y-posy + .5) * r1oSq2Sigma;

        Xexp1 = (X-posx - .5) * r1oSq2Sigma;
        Yexp1 = (Y-posy - .5) * r1oSq2Sigma;

        DeltaX = 0.5 * erf(Xexp0) - 0.5 * erf(Xexp1);
        DeltaY = 0.5 * erf(Yexp0) - 0.5 * erf(Yexp1);
        mu = bg + I0 .* DeltaX .* DeltaY;

        dmu_dx = I0*r1oSq2PiSigma .* ( exp(-Xexp1.*Xexp1) - exp(-Xexp0.*Xexp0)) .* DeltaY;
        dmu_dy = I0*r1oSq2PiSigma .* ( exp(-Yexp1.*Yexp1) - exp(-Yexp0.*Yexp0)) .* DeltaX;
        dmu_dI0 = DeltaX.*DeltaY;
        dmu_dIbg = 1;

        f = smpimg ./ mu - 1;
        dL_dx = dmu_dx .* f;
        dL_dy = dmu_dy .* f;
        dL_dI0 = dmu_dI0 .* f;
        dL_dIbg = dmu_dIbg .* f;

        d2mu_dx = I0*r1oSq2PiSigma3 * ( (X - posx - .5) .* exp(-Xexp1.*Xexp1) - (X - posx + .5) .* exp(-Xexp0.^2) ) .* DeltaY;
        d2mu_dy = I0*r1oSq2PiSigma3 * ( (Y - posy - .5) .* exp(-Yexp1.*Yexp1) - (Y - posy + .5) .* exp(-Yexp0.^2) ) .* DeltaX;
        dL2_dx = d2mu_dx .* f - dmu_dx.^2 .* smpimg ./ (mu.^2);
        dL2_dy = d2mu_dy .* f - dmu_dy.^2 .* smpimg ./ (mu.^2);
        dL2_dI0 = -dmu_dI0.*dmu_dI0 .* smpimg ./ mu.^2;
        dL2_dIbg = -smpimg ./ mu.^2;
        
		posx = posx - sumel(dL_dx) / sumel(dL2_dx);
		posy = posy - sumel(dL_dy) / sumel(dL2_dy);
		I0 = I0 - sumel(dL_dI0) / sumel(dL2_dI0);
		bg = bg - sumel(dL_dIbg) / sumel(dL2_dIbg);
    end
    
    Pestimate = [ posx posy sigma I0 bg ];
end

function Ifisher = compute_fisher(imgsize, P)
    width = imgsize(2);
    height = imgsize(1);

    posx = P(1);
    posy = P(2);
	I0 = P(4);
	bg = P(5);
    sigma = P(3);

	r1oSq2Sigma = 1.0 / (sqrt(2) * sigma);
	r1oSq2PiSigma = 1.0 / (sqrt(2*pi) * sigma);
    
    [X,Y] = meshgrid(0:width-1,0:height-1);

    Xexp0 = (X-posx + .5) * r1oSq2Sigma;
    Yexp0 = (Y-posy + .5) * r1oSq2Sigma;

    Xexp1 = (X-posx - .5) * r1oSq2Sigma;
    Yexp1 = (Y-posy - .5) * r1oSq2Sigma;

    DeltaX = 0.5 * erf(Xexp0) - 0.5 * erf(Xexp1);
    DeltaY = 0.5 * erf(Yexp0) - 0.5 * erf(Yexp1);
    mu = bg + I0 .* DeltaX .* DeltaY;

    dmu_dx = I0*r1oSq2PiSigma .* ( exp(-Xexp1.*Xexp1) - exp(-Xexp0.*Xexp0)) .* DeltaY;
    dmu_dy = I0*r1oSq2PiSigma .* ( exp(-Yexp1.*Yexp1) - exp(-Yexp0.*Yexp0)) .* DeltaX;
    dmu_dI0 = DeltaX.*DeltaY;
    
    Ixx = sumel( dmu_dx .^2 ./ mu );
    Iyy = sumel( dmu_dy .^2 ./ mu );
    Ixy = sumel( dmu_dx .* dmu_dy ./ mu );
    Ixi = sumel ( dmu_dx .* dmu_dI0 ./ mu );
    Iyi = sumel ( dmu_dy .* dmu_dI0 ./ mu );
    Ixbg = sumel ( dmu_dx ./ mu );
    Iybg = sumel ( dmu_dy ./ mu );
    Iii = sumel ( dmu_dI0 .^ 2 ./ mu );
    Iibg = sumel ( dmu_dI0 ./ mu );
    Ibgbg = sumel ( 1./mu.^2 );
    
    Ifisher = [ Ixx Ixy Ixi Ixbg ; Ixy Iyy Iyi Iybg ; Ixi Iyi Iii Iibg ; Ixbg Iybg Iyi Ibgbg ];
end





function Pestimate = fitgaussloop(smpimg, P, iterations)

    width =  size(smpimg,2);
    height =size(smpimg,1);
    
    mean_img = mean(smpimg(:));
	I0 = mean_img*0.5*numel(smpimg);
	bg = mean_img*0.5;
    
    sigma = P(3);

	r1oSq2Sigma = 1.0 / (sqrt(2) * sigma);
	r1oSq2PiSigma = 1.0 / (sqrt(2*pi) * sigma);
	r1oSq2PiSigma3 = 1.0 / (sqrt(2*pi) * sigma^2);

    posx = P(1);
    posy = P(2);
    
	for i=1:iterations
        
		dL_dx = 0.0; 
		dL_dy = 0.0; 
		dL_dI0 = 0.0;
		dL_dIbg = 0.0;
		dL2_dx = 0.0;
		dL2_dy = 0.0;
		dL2_dI0 = 0.0;
		dL2_dIbg = 0.0;

		mu_sum = 0.0;
				
		for y=1:height
			for x=1:width
                Xexp0 = (x-posx + .5) * r1oSq2Sigma;
                Yexp0 = (y-posy + .5) * r1oSq2Sigma;
        
				Xexp1 = (x-posx - .5) * r1oSq2Sigma;
				Yexp1 = (y-posy - .5) * r1oSq2Sigma;
				
				DeltaX = 0.5 * erf(Xexp0) - 0.5 * erf(Xexp1);
				DeltaY = 0.5 * erf(Yexp0) - 0.5 * erf(Yexp1);
				mu = bg + I0 * DeltaX * DeltaY;
				
				dmu_dx = I0*r1oSq2PiSigma * ( exp(-Xexp1*Xexp1) - exp(-Xexp0*Xexp0)) * DeltaY;

				dmu_dy = I0*r1oSq2PiSigma * ( exp(-Yexp1*Yexp1) - exp(-Yexp0*Yexp0)) * DeltaX;
				dmu_dI0 = DeltaX*DeltaY;
				dmu_dIbg = 1;
        
				smp = smpimg(y,x);
				f = smp / mu - 1;
				dL_dx = dL_dx + dmu_dx * f;
				dL_dy = dL_dy + dmu_dy * f;
				dL_dI0 = dL_dI0 + dmu_dI0 * f;
				dL_dIbg = dL_dIbg + dmu_dIbg * f;

				d2mu_dx = I0*r1oSq2PiSigma3 * ( (x - posx - .5) * exp (-Xexp1*Xexp1) - (x - posx + .5) * exp(-Xexp0*Xexp0) ) * DeltaY;
				d2mu_dy = I0*r1oSq2PiSigma3 * ( (y - posy - .5) * exp (-Yexp1*Yexp1) - (y - posy + .5) * exp(-Yexp0*Yexp0) ) * DeltaX;
				dL2_dx = dL2_dx + d2mu_dx * f - dmu_dx*dmu_dx * smp / (mu*mu);
				dL2_dy = dL2_dy + d2mu_dy * f - dmu_dy*dmu_dy * smp / (mu*mu);
				dL2_dI0 = dL2_dI0 -dmu_dI0*dmu_dI0 * smp / (mu*mu);
				dL2_dIbg = dL2_dIbg -smp / (mu*mu);

				mu_sum = mu_sum + mu;
            end
        end
        
		mean_mu = mu_sum / (width*height);
		posx = posx - dL_dx / dL2_dx;
		posy = posy - dL_dy / dL2_dy;
		I0 = I0 - dL_dI0 / dL2_dI0;
		bg = bg - dL_dIbg / dL2_dIbg;
    end
    
    Pestimate = [ posx posy sigma I0 bg ];
end

function [img, imgcv] = makesample(Size, P)
    [Y,X] = ndgrid (0:Size(1)-1, 0:Size(2)-1);
    
    % Read parameters
    Sx = P(1); Sy = P(2); Sigma = P(3); I0 = P(4); Ibg = P(5);
    
    % Center value:
    imgcv = Ibg + I0 * exp ( - ((X-Sx).^2+(Y-Sy).^2) / (2*Sigma^2) ) / (2*pi*Sigma^2);
    imgcv = poissrnd(imgcv);
    
    % Expected values
    edenom = 1 / sqrt(2*Sigma^2);
    DeltaX = 0.5 * erf( (X-Sx + .5) * edenom ) - 0.5 * erf((X-Sx - .5) * edenom);
    DeltaY = 0.5 * erf( (Y-Sy + .5) * edenom ) - 0.5 * erf((Y-Sy - .5) * edenom);
    img = Ibg + I0 * DeltaX .* DeltaY;
end



function d = normalize(d)
    minv = min(d(:));
    maxv = max(d(:));
    
    d = (d-minv) ./ (maxv-minv);
end

function s = sumel(x)
    s=sum(x(:));
end


function test()
    % Parameters format: X, Y, Sigma, I_0, I_bg

    W = 30; H = 30;
    Pcenter = [ W/2 H/2 4 10000 100 ];  %
 
    % Localize
    N = 200;
    iterations = 10;
    
    Ifisher = zeros(4);
    err = zeros(N,5);
    
    for k = 1 : N
        P = Pcenter+(rand(1,5)-.5).*[2 2 0 300 5 ];
        Pinitial = P+(rand(1,5)-.5).*[2 2 0 300 5 ];
        
        smp = poissrnd(makesample([H W], P));
        
        Pestimate = fitgauss(smp, Pinitial, iterations);
        Pest_img = normalize(makesample([H W], Pestimate));
        imshow([ normalize(smp) Pest_img]);
        
        err(k,:) = Pestimate-P;
        fprintf('X:%f, Y:%f, I0:%f, Ibg:%f\n',err(k,1),err(k,2),err(k,4),err(k,5));
        
        Ifisher = Ifisher + compute_fisher(size(smp), Pinitial);
    end

    fprintf('Tracking std: x=%f, y=%f, I0=%f\n', std(err(:,1)), std(err(:,2)), std(err(:,4)));
    
    Ifisher = Ifisher ./ N;
    variance = inv(Ifisher);

    fprintf('Fisher std: x=%f, y=%f, I0=%f\n', sqrt(variance(1,1)), sqrt(variance(2,2)),sqrt(variance(3,3)));
end

