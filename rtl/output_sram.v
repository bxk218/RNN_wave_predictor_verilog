module output_sram(
   input clk,
	input [31:0] data_in,  
	input [8:0] address,  
	input write_en,       
	output [31:0] data_out 
    );
	
	reg [31:0] ram [0:499];
	reg [8:0] addr_buff;
	
	always @(posedge clk) begin
	    if (write_en) begin
		    ram[address] <= data_in;
	    end
		addr_buff <= address;
	end

	assign data_out = ram[addr_buff];
endmodule
