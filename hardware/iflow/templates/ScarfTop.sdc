set_units -time ps
create_clock -name scarf_clock -period 1000 [get_ports clock]
set_clock_uncertainty 50 [get_clocks scarf_clock]
set non_clock_inputs [get_ports -filter {direction == input} {io_*}]
set_input_delay 100 -clock scarf_clock $non_clock_inputs
set_output_delay 100 -clock scarf_clock [all_outputs]
set_false_path -from [get_ports reset]
