function [beadx, beady, beadz, timestamps, frameinfo, axisnames] = qtrk_read_bin_trace(filename, frames, beads, refbead)
% Read XYZ traces from a binary trace output produced by TweezerTracker
% [beadx, beady, beadz, timestamps] = qtrk_read_bin_trace(filename, frames, beads, refbead)
%
% Example: qtrk_read_bin_trace('test.bin', 20:100);  % reads frame 20 to 100 for all beads from test.bin
%
% beadx,beady,beadz: XYZ data, beads are mapped to columns, frames are mapped to rows
% beads: List of beads (1-based indexing)
% refbead: Index of reference bead (-1 to disable reference bead subtraction)
    if nargin<4
        refbead=-1; % no reference bead subtraction
    end
    
    [f_nframes, nbeads, f_ninfocol, axisnames, data_offset] = qtrk_sizeof_bin_trace(filename);
    if f_nframes == 0, return, end %something went wrong
    
    fprintf('File %s has %d beads and %d frames.\n', filename, nbeads, f_nframes);
    
    if nargin<3
        beads = 1:nbeads; % load all beads
    end
    
    if nargin<2
        frames=1:f_nframes;
    end
   
    beadx = zeros(length(frames),length(beads));
    beady = beadx;
    beadz = beadx;    
    timestamps = zeros(length(frames),1);
    frameinfo = zeros(length(frames),f_ninfocol);
    
    fid = fopen(filename);

    bytesPerFrame = 4 + 8 + f_ninfocol * 4 + nbeads * 3 * 4;
    lastFrame=-1;
    for k=1:length(frames)
        
        f=frames(k);
        if f~=lastFrame+1
            filepos = data_offset + bytesPerFrame * (f-1);
            fseek(fid, filepos, -1);
        end
        
        frame_id = fread(fid, 1, 'uint32');
        assert(frame_id == f-1);
        timestamps(k) = fread(fid, 1, 'double');
        frameinfo(k, :) = fread(fid, [1 f_ninfocol], 'single');
        
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