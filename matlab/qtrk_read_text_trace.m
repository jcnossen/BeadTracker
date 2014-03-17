function [beadx,beady,beadz,timestamps,frameinfo] = qtrk_read_text_trace(filename, frames, beads, refbead)
    d=dlmread(filename);
        
    [path,name]=fileparts(filename);
    
    motorfile = [path name '_motors.txt'];
    motord = dlmread(motorfile);
    
    nframes=size(d,1);
    nbeads = (size(d,2)-2)/3;
    
    fprintf('%s has %d beads and %d frames\n', filename, nbeads,nframes);
    
    if nargin<4
        refbead=-1;
    end
    if nargin<3 || beads<0
        beads=1:nbeads;
    end
    if nargin<2 || frames<0
        frames=1:nframes;
    end
    beadx = zeros(nframes,length(beads));
    beady=beadx; beadz=beadx;
    frameinfo = motord(frames,3:end);
    timestamps = d(frames,2);
   
    for k=1:beads
        b=beads(k)-1;
        beadx(frames,k) = d(frames, 3+3*b);
        beady(frames,k) = d(frames, 4+3*b);
        beadz(frames,k) = d(frames, 5+3*b);
        
        if refbead>=0
            beadx(frames,k) = beadx(frames,k)-d(frames, 3+3*(refbead-1));
            beady(frames,k) = beady(frames,k)-d(frames, 3+4*(refbead-1));
            beadz(frames,k) = beadz(frames,k)-d(frames, 3+5*(refbead-1));
        end
    end
    
    
end
