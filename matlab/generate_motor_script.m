function txt=generate_motor_script(axis, values, idle_times, motor_wait_time)

    if nargin==0
        axis='magpos';
        values=2:10;
        idle_times=10;
    end
    
    if nargin<4
        motor_wait_time=0.1;
    end
    
    if isscalar(idle_times), 
        idle_times=ones(1,length(values))*idle_times;
    end

    for k=1:length(values)
        lines{k} = sprintf('move %s %.2f; idle %.2f; section; idle %.2f; section;\n', ...
            axis, values(k), motor_wait_time, idle_times(k));
    end
    txt = cell2mat(lines);

    clipboard('copy', txt);

end