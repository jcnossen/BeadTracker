function [beadx, beady, beadz, timestamps, frameinfo, axisnames] = qtrk_read_trace(filename, frames, beads, refbead, oldver)
% Read XYZ traces from a binary trace output produced by TweezerTracker
% [beadx, beady, beadz, timestamps, frameinfo, axisnames] =  ...
%    qtrk_read_bin_trace(filename, frames, beads, refbead)
%
% Example: qtrk_read_bin_trace('test.bin', 20:100);  % reads frame 20 to 100 for all beads from test.bin
%
% beadx,beady,beadz: XYZ data, beads are mapped to columns, frames are mapped to rows
% frames: list of frame numbers, or -1 for all frames
% beads: List of beads (1-based indexing)
% refbead: Index of reference bead (-1 to disable reference bead subtraction)
    if nargin<4
        refbead=-1; % no reference bead subtraction
    end
    
    if nargin<5
        oldver=0;
    end
    
    cachename = sprintf('%s_f%d_b%d_r%d.mat', filename, length(frames), length(beads), refbead(1));
%    if (exist(cachename, 'file'))
 %       d=load(cachename);
    
    
    [~,~,ext]=fileparts(filename);
    if strcmp(ext,'.txt')
        [beadx,beady,beadz,timestamps,frameinfo] = qtrk_read_text_trace(filename, frames, beads, refbead);
        axisnames={'unknown'};
    else
        [beadx,beady,beadz,timestamps,frameinfo,axisnames] = qtrk_read_bin_trace(filename,frames, beads, refbead, oldver);
    end
    
    
end
