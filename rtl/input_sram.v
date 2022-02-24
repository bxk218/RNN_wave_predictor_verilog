module input_sram
#( parameter DATA_WIDTH = 16, parameter ADDR_DEPTH = 9, parameter MAX_ADDR = 499)
(
   input clk,
	input rst_n,
	input data_valid,
	input read_start,
	input yhat_valid,
	input int_clear,
	input [DATA_WIDTH-1:0] data_in,  
	output reg datafeed_en,
	output reg complete,
	output [DATA_WIDTH-1:0] data_out 
);
	
	parameter [2:0]	IDLE			= 3'b000,
			  				DATA_WR		= 3'b001,
							RW_BRANCH	= 3'b010,
							READ_START	= 3'b011,
							DATA_RD		= 3'b100,
							DATA_FEED	= 3'b101,
							COMPLETE		= 3'b110,
							INT_CLEAR	= 3'b111;
	reg [2:0] state;
	reg [2:0] next_state;

	reg [DATA_WIDTH-1:0] ram [0:MAX_ADDR];
	reg [ADDR_DEPTH-1:0] addr_buff;

// Address and WREN generator
	reg [ADDR_DEPTH-1:0] address;
	reg write_en;	
	reg write_en_t;	
	reg read_en;	
	reg read_en_t;	
	reg address_inc;	
	reg address_inc_t;	
	reg address_clear;	
	reg address_clear_t;	
	reg datafeed_en_t;
	reg complete_t;

// State machine
  	always @(*) begin
	   case (state)
			IDLE: begin
		 		next_state = DATA_WR;
				write_en_t = 0;
				read_en_t = 0;
				address_inc_t = 0;
				address_clear_t = 0;
				datafeed_en_t = 0;
				complete_t = 0;
			end
			DATA_WR: begin
				if (!data_valid) begin 
					next_state = DATA_WR;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 0;
					address_clear_t = 0;
					datafeed_en_t = 0;
					complete_t = 0;
		 		end 
				else begin
					next_state = RW_BRANCH;
					write_en_t = 1;
					read_en_t = 0;
					address_inc_t = 0;
					address_clear_t = 0;
					datafeed_en_t = 0;
					complete_t = 0;
				end
			end 
			RW_BRANCH: begin
				if (address < 499) begin 
					next_state = IDLE;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 1;
					address_clear_t = 0;
					datafeed_en_t = 0;
					complete_t = 0;
		 		end 
				else begin
					next_state = READ_START;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 0;
					address_clear_t = 1;
					datafeed_en_t = 0;
					complete_t = 0;
				end
			end 
			READ_START: begin
				if (!read_start) begin 
		 			next_state = READ_START;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 0;
					address_clear_t = 0;
					datafeed_en_t = 0;
					complete_t = 0;
		 		end 
				else begin
		 			next_state = DATA_RD;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 0;
					address_clear_t = 0;
					datafeed_en_t = 0;
					complete_t = 0;
		 		end 
			end
			DATA_RD: begin
				next_state = DATA_FEED;
				write_en_t = 0;
				read_en_t = 1;
				address_inc_t = 0;
				address_clear_t = 0;
				datafeed_en_t = 0;
				complete_t = 0;
			end 
			DATA_FEED: begin
				if (!yhat_valid) begin 
					next_state = DATA_FEED;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 0;
					address_clear_t = 0;
					datafeed_en_t = 1;
					complete_t = 0;
		 		end 
				else begin
					next_state = COMPLETE;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 0;
					address_clear_t = 0;
					datafeed_en_t = 0;
					complete_t = 0;
				end
			end 
			COMPLETE: begin
				if (address != MAX_ADDR) begin 
					next_state = DATA_RD;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 1;
					address_clear_t = 0;
					datafeed_en_t = 0;
					complete_t = 0;
		 		end 
				else begin
					next_state = INT_CLEAR;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 0;
					address_clear_t = 0;
					datafeed_en_t = 0;
					complete_t = 1;
				end
			end 
			INT_CLEAR: begin
				if (!int_clear) begin 
					next_state = INT_CLEAR;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 0;
					address_clear_t = 0;
					datafeed_en_t = 0;
					complete_t = 1;
		 		end 
				else begin
					next_state = IDLE;
					write_en_t = 0;
					read_en_t = 0;
					address_inc_t = 0;
					address_clear_t = 0;
					datafeed_en_t = 0;
					complete_t = 0;
				end
			end 
			default: next_state = IDLE; 
		endcase
	end

	// latchd signals for state machines
	always @(posedge clk or negedge rst_n) begin
	  if(!rst_n) begin
			state <= IDLE;
			write_en <= 0;
			address_inc <= 0;
			address_clear <= 0;
			datafeed_en <= 0;
			complete <= 0;
			read_en <= 0;
		end
		else begin
			state <= next_state;
			write_en <= write_en_t;
			address_inc <= address_inc_t;
			address_clear <= address_clear_t;
			datafeed_en <= datafeed_en_t;
			complete <= complete_t;
			read_en <= read_en_t;
		end
	end

	always @(posedge clk or negedge rst_n) begin
	  	if(!rst_n) begin
			address <= 0;
		end
		else begin
			if (address_clear) address <= 0;
			else begin
				if (address_inc) address <= address + 1;
				else address <= address;
			end
		end
	end

	always @(posedge clk) begin
	    if (write_en) begin
		    ram[address] <= data_in;
	    end
		 addr_buff <= address;
	end

	assign data_out = ram[addr_buff];
endmodule
