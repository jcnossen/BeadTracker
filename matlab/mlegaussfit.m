% 2D Gaussian fit implementation using Maximum Likelihood Method described by 
%   "Fast, single-molecule localization that achieves theoretically
%   minimum uncertainty" CS Smith, N Joseph, B Rieger, KA Lidke - Nature methods, 2010
%
% Implementation by Jelmer Cnossen
%
% Function handles for the various functions will be returned if you call
% this file.
%
% Syntax:
%   estimate = g.fit(image, sigma, initial_guess, iterations(  optional ) )
%   fisher_matrix = g.compute_fisher([Height Width], sigma, parameters)
%   sample_image = g.sample(image_size, sigma, parameters);
%
% Parameter format:
%   [ X Y SpotIntensity BackgroundIntensityPerPixel ]
%
% Example:
%   g = mlegaussfit();
%   sigma = 5;
%   smp = poissrnd(g.sample([ 30 30 ], sigma, [15 15 2000 10]));
%   estimate = g.fit(smp, sigma, [ 10 10 1000 10 ])
function functions = mlegaussfit()    
    functions.test = @test;
    functions.fit = @fitgauss;
    functions.compute_fisher = @compute_fisher;
    functions.sample = @makesample;
    functions.show = @(img) imshow(normalize(img));
end

function estimate = fitgauss(smpimg, sigma, initial, iterations)
    
    width = size(smpimg,2);
    height =size(smpimg,1);

    if nargin<4
        iterations = 20;
    end
    
    if nargin<2
        posx = initial(1);
        posy = initial(2);
    else 
        posx = width/2;
        posy = height/2;
    end

    mean_img = mean(smpimg(:));
	I0 = mean_img*0.5*numel(smpimg);
	bg = mean_img*0.5;

	r1oSq2Sigma = 1.0 / (sqrt(2) * sigma);
	r1oSq2PiSigma = 1.0 / (sqrt(2*pi) * sigma);
	r1oSq2PiSigma3 = 1.0 / (sqrt(2*pi) * sigma^2);
    
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
        
        fprintf('[%d] I0: %f\n', i, I0);
    end
    
    estimate = [ posx posy I0 bg ];
end

function Ifisher = compute_fisher(imgsize, sigma, P)
    width = imgsize(2);
    height = imgsize(1);

    posx = P(1);
    posy = P(2);
	I0 = P(3);
	bg = P(4);

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



function Ifisher = compute_fisher_numerical(imgsize, sigma, P)
    sample = @(par) makesample(imgsize, sigma, par);

    mu = sample (P );
    d=0.001;
    dmu_dx = (sample(P + [ d 0 0 0 ]) - sample(P - [ d 0 0 0 ]))/(2*d);
    dmu_dy = (sample(P + [ 0 d 0 0 ]) - sample(P - [ 0 d 0 0 ]))/(2*d);
    dmu_dI0 = (sample(P + [ 0 0 d 0 ]) - sample(P - [ 0 0 d 0 ]))/(2*d);
    %dmu_dbg = (sample(P + [ 0 0 0 d ]) - sample(P - [ 0 0 0 d ]))/(2*d);
    
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



function img = makesample(Size, sigma, P)
    [Y,X] = ndgrid (0:Size(1)-1, 0:Size(2)-1);
    
    % Read parameters
    Sx = P(1); Sy = P(2); I0 = P(3); Ibg = P(4);
    
    % Expected values
    edenom = 1 / sqrt(2*sigma^2);
    DeltaX = 0.5 * erf( (X-Sx + .5) * edenom ) - 0.5 * erf((X-Sx - .5) * edenom);
    DeltaY = 0.5 * erf( (Y-Sy + .5) * edenom ) - 0.5 * erf((Y-Sy - .5) * edenom);
    img = Ibg + I0 * DeltaX .* DeltaY;
end


function imgcv = makesamplecv(Size, sigma, P)
    [Y,X] = ndgrid (0:Size(1)-1, 0:Size(2)-1);
    
    % Read parameters
    Sx = P(1); Sy = P(2); I0 = P(3); Ibg = P(4);
    
    % Center value:
    imgcv = Ibg + I0 * exp ( - ((X-Sx).^2+(Y-Sy).^2) / (2*sigma^2) ) / (2*pi*sigma^2);
end



function d = normalize(d)
    d=double(d);
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
    sigma = 4;
    Pcenter = [ W/2 H/2 10000 100 ];  %
 
    % Localize
    N = 20;
    iterations = 10;
    
    Ifisher = zeros(4);
    Ifisher_n = Ifisher;
    err = zeros(N,4);
    
    for k = 1 : N
        P = Pcenter+(rand(1,4)-.5).*[2 2 300 5 ];
        Pinitial = P+(rand(1,4)-.5).*[2 2 300 5 ];
        
        smp = poissrnd(makesample([H W], sigma, P ));
        
        Pestimate = fitgauss(smp, sigma, Pinitial, iterations);
        Pest_img = normalize(makesample([H W], sigma, Pestimate));
        imshow([ normalize(smp) Pest_img]);
        
        err(k,:) = Pestimate-P;
        fprintf('X:%f, Y:%f, I0:%f, Ibg:%f\n',err(k,1),err(k,2),err(k,3),err(k,4));
        
        Ifisher = Ifisher + compute_fisher(size(smp), sigma, Pinitial);
        Ifisher_n = Ifisher_n + compute_fisher_numerical(size(smp), sigma, Pinitial);
        
    end

    fprintf('Tracking std: x=%f, y=%f, I0=%f\n', std(err(:,1)), std(err(:,2)), std(err(:,3)));
    
    variance = inv(Ifisher ./ N);
    variance_n = inv(Ifisher_n ./ N);

    fprintf('Fisher std: x=%f, y=%f, I0=%f\n', sqrt(variance(1,1)), sqrt(variance(2,2)),sqrt(variance(3,3)));
    fprintf('Numeric Fisher std: x=%f, y=%f, I0=%f\n', sqrt(variance_n(1,1)), sqrt(variance_n(2,2)),sqrt(variance_n(3,3)));
    
end

