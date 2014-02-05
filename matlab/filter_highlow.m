
function [high, low] = filter_highlow(d, fps, cutoff)

    if (nargin<3)
        d=randn(1,100)+5*cos( (1:100) / 100 * 2 * pi * 5 );
        fps=100;
        cutoff=10;
    end

    N = length(d);
    d = reshape(d, 1, N);
    fd = fft(d);
    
    timestep = 1/fps;
    freqstep = 1/(timestep * N);
    freq = -0.5/timestep + freqstep/2 : freqstep : 0.5/timestep - freqstep/2;
    
    
    fd_low =  fd .* fftshift( abs(freq) <=cutoff);
    fd_high = fd .* fftshift(abs(freq)>=cutoff);
    
    high = real(ifft(fd_high));
    low = real(ifft(fd_low));
    
    if nargin<3, 
        subplot(411); plot(abs(fd));
        subplot(412); plot(high);
        subplot(413); plot(low);
        subplot(414); plot(d);
    end;
    
end
