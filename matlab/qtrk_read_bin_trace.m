function [beadx, beady, beadz, timestamps] = qtrk_read_bin_trace(filename, frames, beads, refbead)
% Read XYZ traces from a binary trace output produced by TweezerTracker
% [beadx, beady, beadz, timestamps] = qtrk_read_bin_trace(filename, frames, beads, refbead)
% beadx,beady,beadz: XYZ data, beads are mapped to columns, frames are mapped to rows
% beads: List of beads (1-based indexing)
% refbead: Index of reference bead (-1 to disable reference bead subtraction)
    if nargin<4
        refbead=-1; % no reference bead subtraction
    end
    
    [f_nframes, f_nbeads] = qtrk_bintrace_size(filename);
    fprintf('File %s has %d beads and %d frames.\n', filename, f_nbeads, f_nframes);
    
    if nargin<3
        beads = 1:f_nbeads; % load all beads
    end
   
    beadx = zeros(length(frames),length(beads));
    beady = beadx;
    beadz = beadx;    
    timestamps = zeros(length(frames),1);
    
    fid = fopen(filename);

    bytesPerFrame = 4 + 8 + f_nbeads * 3 * 4;
    lastFrame=-1;
    for k=1:length(frames)
        
        f=frames(k);
        if f~=lastFrame+1
            filepos = 4+ bytesPerFrame * (f-1);
            fseek(fid, filepos, -1);
        end
        
        frame_id = fread(fid, 1, 'uint32');
        timestamps(k) = fread(fid, 1, 'double');
        xyz = fread(fid, nbeads * 3, 'single');
        
        beadx(k, :) = xyz( (0:nbeads-1) * 3 + 1 );
        beady(k, :) = xyz( (0:nbeads-1) * 3 + 2 );
        beadz(k, :) = xyz( (0:nbeads-1) * 3 + 3 );
        
        if refbead >= 0
            beadx(k, :) = beadx(k,:) - beadx(k, refbead);
            beady(k, :) = beady(k,:) - beady(k, refbead);
            beadz(k, :) = beadz(k,:) - beadz(k, refbead);
        end
        
        lastFrame=f;
    end
    
    fclose(fid);

end