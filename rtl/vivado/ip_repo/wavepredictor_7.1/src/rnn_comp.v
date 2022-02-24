module rnn_comp
#( parameter DATA_WIDTH = 16, parameter PRECISION = 128, parameter N_HIDDEN = 9, parameter SFT_R = 7)
(	input clk, 									// Clock
	input rst_n,								// Reset 
	input data_valid, 						// Input data valid signal - input is fetched when it is 1 
	input [DATA_WIDTH-1:0] data_in, 		// Input data
	input read_start,  						// All input data are buffered and ready to compute
	input int_clear,  						// Interrupt clear signal - for future use for interrupt handling
	input [8:0] read_address,				// Read address for predicted values of RNN
	output complete,							// Complete the computation and ready to read the final results
	output [DATA_WIDTH*2-1:0] output_mem		// Prediction results
); 

//--------------------------------------------------------------------------
//				State variables for RNN computation sequences
//--------------------------------------------------------------------------
	parameter [2:0] 	IDLE			= 3'b000,
			  				START			= 3'b001,
							MATMULT_EN 	= 3'b010,
							TANH_EN 		= 3'b011,
							PRED_EN 		= 3'b100,
							PRED_DONE	= 3'b101;
	reg [2:0] state;
	reg [2:0] next_state;

	wire signed [DATA_WIDTH-1:0] sram_data_in;
	wire signed [DATA_WIDTH-1:0] data;

//--------------------------------------------------------------------------
//		* RNN Model Coefficients
//			- Wih:	Weight for input layer
//			- bih: 	Bias for input layer
//			- Whh:	Weight for hidden layer 
//			- bhh: 	Bias for hidden layer
//			- Who:	Weight for output layer 
//			- bho: 	Bias for output layer
//--------------------------------------------------------------------------
	reg signed [DATA_WIDTH-1:0] Wih [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH-1:0] Whh [0:N_HIDDEN-1][0:N_HIDDEN-1];
	reg signed [DATA_WIDTH-1:0] Who [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH-1:0] bih [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH-1:0] bhh [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH-1:0] bho;

//--------------------------------------------------------------------------
//		* Matrix multiplication for Input layer (Wih x Xi + bih)	
//			- Wi:	Weight for input vectors
//			- Xi: Input vectors
//			- bi: Bias for input vectors
//--------------------------------------------------------------------------
	reg signed [DATA_WIDTH*2-1:0] WiXi [0:N_HIDDEN-1];			//Wih@data
	reg signed [DATA_WIDTH*2-1:0] WiXibi [0:N_HIDDEN-1];		//Wih@data + bhi
	reg signed [DATA_WIDTH*2-1:0] WiXibi_d [0:N_HIDDEN-1];	//Wih@data + bhi

	wire signed [DATA_WIDTH-1:0] WiXibi_out0_t;
	wire signed [DATA_WIDTH-1:0] WiXibi_out1_t;
	wire signed [DATA_WIDTH-1:0] WiXibi_out2_t;
	wire signed [DATA_WIDTH-1:0] WiXibi_out3_t;
	wire signed [DATA_WIDTH-1:0] WiXibi_out4_t;
	wire signed [DATA_WIDTH-1:0] WiXibi_out5_t;
	wire signed [DATA_WIDTH-1:0] WiXibi_out6_t;
	wire signed [DATA_WIDTH-1:0] WiXibi_out7_t;
	wire signed [DATA_WIDTH-1:0] WiXibi_out8_t;

	wire [DATA_WIDTH-1:0] WiXibi_out0;
	wire [DATA_WIDTH-1:0] WiXibi_out1;
	wire [DATA_WIDTH-1:0] WiXibi_out2;
	wire [DATA_WIDTH-1:0] WiXibi_out3;
	wire [DATA_WIDTH-1:0] WiXibi_out4;
	wire [DATA_WIDTH-1:0] WiXibi_out5;
	wire [DATA_WIDTH-1:0] WiXibi_out6;
	wire [DATA_WIDTH-1:0] WiXibi_out7;
	wire [DATA_WIDTH-1:0] WiXibi_out8;

//--------------------------------------------------------------------------
//		* Matrix multiplication for hidden layer (Whh x Hi + bhh)	
//			- Wi:	Weight for input vectors
//			- Xi: Input vectors
//			- bi: Bias for input vectors
//--------------------------------------------------------------------------
	reg signed [DATA_WIDTH*2-1:0] WhHi [0:N_HIDDEN-1];				//Whh@hhi(t-1)
	reg signed [DATA_WIDTH*2-1:0] WhHi_init [0:N_HIDDEN-1];		//Whh@hhi (init = 0)
	reg signed [DATA_WIDTH*2-1:0] WhHibh [0:N_HIDDEN-1];			//Whh@hhi(t-1) + bhh 
	reg signed [DATA_WIDTH*2-1:0] WhHibh_d [0:N_HIDDEN-1];		//Clocked Whh@hhi(t-1) + bhh
	reg signed [DATA_WIDTH*2-1:0] WhHibh_prev [0:N_HIDDEN-1];	//Whh@hhi(t-1) + bhh (init = 0)

	reg signed [DATA_WIDTH*2-1:0] WhHi0 [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH*2-1:0] WhHi1 [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH*2-1:0] WhHi2 [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH*2-1:0] WhHi3 [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH*2-1:0] WhHi4 [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH*2-1:0] WhHi5 [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH*2-1:0] WhHi6 [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH*2-1:0] WhHi7 [0:N_HIDDEN-1];
	reg signed [DATA_WIDTH*2-1:0] WhHi8 [0:N_HIDDEN-1];

	wire signed [DATA_WIDTH-1:0] WhHibh_out0_t;
	wire signed [DATA_WIDTH-1:0] WhHibh_out1_t;
	wire signed [DATA_WIDTH-1:0] WhHibh_out2_t;
	wire signed [DATA_WIDTH-1:0] WhHibh_out3_t;
	wire signed [DATA_WIDTH-1:0] WhHibh_out4_t;
	wire signed [DATA_WIDTH-1:0] WhHibh_out5_t;
	wire signed [DATA_WIDTH-1:0] WhHibh_out6_t;
	wire signed [DATA_WIDTH-1:0] WhHibh_out7_t;
	wire signed [DATA_WIDTH-1:0] WhHibh_out8_t;

	wire [DATA_WIDTH-1:0] WhHibh_out0;
	wire [DATA_WIDTH-1:0] WhHibh_out1;
	wire [DATA_WIDTH-1:0] WhHibh_out2;
	wire [DATA_WIDTH-1:0] WhHibh_out3;
	wire [DATA_WIDTH-1:0] WhHibh_out4;
	wire [DATA_WIDTH-1:0] WhHibh_out5;
	wire [DATA_WIDTH-1:0] WhHibh_out6;
	wire [DATA_WIDTH-1:0] WhHibh_out7;
	wire [DATA_WIDTH-1:0] WhHibh_out8;

	reg signed [DATA_WIDTH*2-1:0] WhHi0_SUM;
	reg signed [DATA_WIDTH*2-1:0] WhHi1_SUM;
	reg signed [DATA_WIDTH*2-1:0] WhHi2_SUM;
	reg signed [DATA_WIDTH*2-1:0] WhHi3_SUM;
	reg signed [DATA_WIDTH*2-1:0] WhHi4_SUM;
	reg signed [DATA_WIDTH*2-1:0] WhHi5_SUM;
	reg signed [DATA_WIDTH*2-1:0] WhHi6_SUM;
	reg signed [DATA_WIDTH*2-1:0] WhHi7_SUM;
	reg signed [DATA_WIDTH*2-1:0] WhHi8_SUM;

	wire signed [DATA_WIDTH-1:0] Hi0;
	wire signed [DATA_WIDTH-1:0] Hi1;
	wire signed [DATA_WIDTH-1:0] Hi2;
	wire signed [DATA_WIDTH-1:0] Hi3;
	wire signed [DATA_WIDTH-1:0] Hi4;
	wire signed [DATA_WIDTH-1:0] Hi5;
	wire signed [DATA_WIDTH-1:0] Hi6;
	wire signed [DATA_WIDTH-1:0] Hi7;
	wire signed [DATA_WIDTH-1:0] Hi8;

//--------------------------------------------------------------------------
//		* Matrix multiplication for output layer (Who x Ho + bho)	
//			- Who:	Weight for output vectors
//			- Ho: 	output vectors from hidden layer
//			- bho: 	Bias for output vectors
//--------------------------------------------------------------------------
	reg signed [DATA_WIDTH*2-1:0] output_mem_d;				// Prediction results
	wire signed [DATA_WIDTH*2-1:0] output_mem_t;				

	reg signed [DATA_WIDTH-1:0] Ho0;								// Hidden layer output = output layer input
	reg signed [DATA_WIDTH-1:0] Ho1;
	reg signed [DATA_WIDTH-1:0] Ho2;
	reg signed [DATA_WIDTH-1:0] Ho3;
	reg signed [DATA_WIDTH-1:0] Ho4;
	reg signed [DATA_WIDTH-1:0] Ho5;
	reg signed [DATA_WIDTH-1:0] Ho6;
	reg signed [DATA_WIDTH-1:0] Ho7;
	reg signed [DATA_WIDTH-1:0] Ho8;

	wire signed [DATA_WIDTH-1:0] Ho0_t;
	wire signed [DATA_WIDTH-1:0] Ho1_t;
	wire signed [DATA_WIDTH-1:0] Ho2_t;
	wire signed [DATA_WIDTH-1:0] Ho3_t;
	wire signed [DATA_WIDTH-1:0] Ho4_t;
	wire signed [DATA_WIDTH-1:0] Ho5_t;
	wire signed [DATA_WIDTH-1:0] Ho6_t;
	wire signed [DATA_WIDTH-1:0] Ho7_t;
	wire signed [DATA_WIDTH-1:0] Ho8_t;

	reg signed [DATA_WIDTH*2-1:0] yhat;							// Who@hho + bho
	reg signed [DATA_WIDTH*2-1:0] yhat_d;						// Clocked Who@hho + bho
	reg signed [DATA_WIDTH*2-1:0] yhat0_t;						
	reg signed [DATA_WIDTH*2-1:0] yhat1_t;						
	reg signed [DATA_WIDTH*2-1:0] yhat2_t;					
	reg signed [DATA_WIDTH*2-1:0] yhat3_t;					
	reg signed [DATA_WIDTH*2-1:0] yhat4_t;				
	reg signed [DATA_WIDTH*2-1:0] yhat5_t;						
	reg signed [DATA_WIDTH*2-1:0] yhat6_t;						
	reg signed [DATA_WIDTH*2-1:0] yhat7_t;						
	reg signed [DATA_WIDTH*2-1:0] yhat8_t;						

//--------------------------------------------------------------------------
//		All inputs and output for control state machine
//--------------------------------------------------------------------------
	reg [3:0] scheduler;												
	reg yhat_valid;
	wire datafeed_en;
	reg Hi_init;
	reg scheduler_en;
	reg matmult_en;
	reg tanh_en;
	reg pred_en; 
	reg scheduler_en_d;
	reg matmult_en_d;
	reg tanh_en_d;
	reg pred_en_d; 
	reg matmult_done;
	reg tanh_done;
	reg pred_done; 
	wire matmult_done_t;
	wire tanh_done_t;
	wire pred_done_t; 
	reg yhat_valid_t; 
	reg address_inc;	
	reg address_inc_t;	
	reg complete_t;	

	reg [8:0] address;
	integer i,j;

//--------------------------------------------------------------------------
//		Instantiation for activation function (tanh)
//--------------------------------------------------------------------------
	tanh TANH0 (.X(Hi0), .Y(Ho0_t));
	tanh TANH1 (.X(Hi1), .Y(Ho1_t));
	tanh TANH2 (.X(Hi2), .Y(Ho2_t));
	tanh TANH3 (.X(Hi3), .Y(Ho3_t));
	tanh TANH4 (.X(Hi4), .Y(Ho4_t));
	tanh TANH5 (.X(Hi5), .Y(Ho5_t));
	tanh TANH6 (.X(Hi6), .Y(Ho6_t));
	tanh TANH7 (.X(Hi7), .Y(Ho7_t));
	tanh TANH8 (.X(Hi8), .Y(Ho8_t));

//--------------------------------------------------------------------------
//		Instantiation for input memory
//--------------------------------------------------------------------------
	input_sram #( .DATA_WIDTH(DATA_WIDTH), .ADDR_DEPTH(9), .MAX_ADDR(499)) INPUT_SRAM
	(
   	.clk					(clk),
		.rst_n				(rst_n),
		.data_valid			(data_valid),
		.read_start			(read_start),
		.yhat_valid			(yhat_valid),
		.int_clear			(int_clear),
		.data_in				(data_in),  
		.datafeed_en		(datafeed_en),
		.complete			(complete),
		.data_out			(sram_data_in) 
	);

//--------------------------------------------------------------------------
//		Instantiation for input memory
//--------------------------------------------------------------------------
	output_sram OUTPUT_SRAM
	(
		.clk					(clk),
		.data_in				(yhat_d),  
		.address				(address),  
		.write_en			(yhat_valid),       
		.data_out			(output_mem_t) 
	);

//--------------------------------------------------------------------------
//		Control State Machine blocks
//--------------------------------------------------------------------------
	always @(state or datafeed_en or matmult_done or tanh_done or pred_done) begin
	   case (state)
			IDLE: begin
		 		next_state = START;
				scheduler_en = 0;
				matmult_en = 0;
				tanh_en = 0;
				pred_en = 0;
				yhat_valid_t = 0;
				address_inc_t = 0;
			end
			START: begin
				if (!datafeed_en) begin 
					next_state = START;
					scheduler_en = 0;
					matmult_en = 0;
					tanh_en = 0;
					pred_en = 0;
					yhat_valid_t = 0;
					address_inc_t = 0;
		 		end 
				else begin
					next_state = MATMULT_EN;
					scheduler_en = 1;
					matmult_en = 1;
					tanh_en = 0;
					pred_en = 0;
					yhat_valid_t = 0;
					address_inc_t = 0;
				end
			end 
			MATMULT_EN: begin
				if (!matmult_done) begin 
					next_state = MATMULT_EN;
					scheduler_en = 1;
					matmult_en = 1;
					tanh_en = 0;
					pred_en = 0;
					yhat_valid_t = 0;
					address_inc_t = 0;
		 		end 
				else begin
					next_state = TANH_EN;
					scheduler_en = 1;
					matmult_en = 0;
					tanh_en = 1;
					pred_en = 0;
					yhat_valid_t = 0;
					address_inc_t = 0;
				end
			end 
			TANH_EN: begin
				if (!tanh_done) begin 
					next_state = TANH_EN;
					scheduler_en = 1;
					matmult_en = 0;
					tanh_en = 1;
					pred_en = 0;
					yhat_valid_t = 0;
					address_inc_t = 0;
		 		end 
				else begin
					next_state = PRED_EN;
					scheduler_en = 1;
					matmult_en = 0;
					tanh_en = 0;
					pred_en = 1;
					yhat_valid_t = 0;
					address_inc_t = 0;
				end
			end 
			PRED_EN: begin
				if (!pred_done) begin 
					next_state = PRED_EN;
					scheduler_en = 1;
					matmult_en = 0;
					tanh_en = 0;
					pred_en = 1;
					yhat_valid_t = 0;
					address_inc_t = 0;
		 		end 
				else begin
					next_state = PRED_DONE;
					scheduler_en = 1;
					matmult_en = 0;
					tanh_en = 0;
					pred_en = 1;
					yhat_valid_t = 1;
					address_inc_t = 0;
				end
			end 
			PRED_DONE: begin
				next_state = IDLE;
				scheduler_en = 1;
				matmult_en = 0;
				tanh_en = 0;
				pred_en = 0;
				yhat_valid_t = 0;
				address_inc_t = 1;
			end 
			default: next_state = IDLE; 
		endcase
	end

//--------------------------------------------------------------------------
//		All input/output signals for control state machine
//--------------------------------------------------------------------------
	always @(posedge clk or negedge rst_n) begin
	  if(!rst_n) begin
			state <= IDLE;
			scheduler_en_d <= 0;
			matmult_en_d <= 0;
			tanh_en_d <= 0;
			pred_en_d <= 0;
			yhat_valid <= 0;
			address_inc <= 0;
			complete_t <= 0;
			output_mem_d <= 0;
		end
		else begin
			state <= next_state;
			scheduler_en_d <= scheduler_en;
			matmult_en_d <= matmult_en;
			tanh_en_d <= tanh_en;
			pred_en_d <= pred_en;
			yhat_valid <= yhat_valid_t;
			address_inc <= address_inc_t;
			complete_t <= complete;
			output_mem_d <= output_mem_t;
		end
	end

//--------------------------------------------------------------------------
//		Output memory address generation for storing prediction results
//--------------------------------------------------------------------------
	always @(posedge clk or negedge rst_n) begin
	  	if(!rst_n) begin
			address <= 0;
		end
		else begin
			if (complete_t) address <= read_address;
			else begin
				if (address_inc) address <= address + 1;
				else address <= address;
			end
		end
	end
	
//--------------------------------------------------------------------------
//		Timing tuning for variables for state transition
//--------------------------------------------------------------------------
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			Hi_init <= 0;
			scheduler <= 0;
			matmult_done <= 0;
			tanh_done <= 0;
			pred_done <= 0;
			for (i=0; i<N_HIDDEN; i=i+1) begin
				WhHibh_prev[i] <= 0;
			end
		end
		else if (scheduler_en_d) begin
			if (yhat_valid)
				scheduler <= 0;
			else
				scheduler <= scheduler + 1;
			matmult_done <= matmult_done_t;
			tanh_done <= tanh_done_t;
			pred_done <= pred_done_t;
			if (pred_done) Hi_init <= 1;
			if (matmult_done) begin
				for (i=0; i<N_HIDDEN; i=i+1) begin
					WhHibh_prev[i] <= WhHibh[i];
				end
 			end
		end
	end

//--------------------------------------------------------------------------
//		Matrix multiplication: Wih*Xi+bih / Whh*Hi+bhh / Who*Ho+bho
//--------------------------------------------------------------------------
	always @(*) begin
		
//    All coefficients
	   Wih[0] <= 16'b1111111111000001; //-63
   	Wih[1] <= 16'b0000000000001101; //13
   	Wih[2] <= 16'b0000000000000101; //5
   	Wih[3] <= 16'b0000000001010000; //80
   	Wih[4] <= 16'b0000000000011001; //25
   	Wih[5] <= 16'b0000000000110011; //51
   	Wih[6] <= 16'b0000000001001011; //75
   	Wih[7] <= 16'b0000000000100110; //38
   	Wih[8] <= 16'b0000000000001111; //15

	   Whh[0][0] <= 16'b1111111111101101; //-19
	   Whh[0][1] <= 16'b0000000000011111; //31
	   Whh[0][2] <= 16'b1111111111010101; //-43
	   Whh[0][3] <= 16'b1111111111110111; //-9
	   Whh[0][4] <= 16'b0000000000001100; //12
	   Whh[0][5] <= 16'b0000000000001110; //14
	   Whh[0][6] <= 16'b0000000000000000; //0
	   Whh[0][7] <= 16'b0000000000001100; //12
	   Whh[0][8] <= 16'b0000000000010101; //21
	   Whh[1][0] <= 16'b1111111111100110; //-26
	   Whh[1][1] <= 16'b1111111111100111; //-25
	   Whh[1][2] <= 16'b1111111111011100; //-36
	   Whh[1][3] <= 16'b1111111111100001; //-31
	   Whh[1][4] <= 16'b0000000000001111; //15
	   Whh[1][5] <= 16'b0000000000010100; //20
	   Whh[1][6] <= 16'b1111111111110110; //-10
	   Whh[1][7] <= 16'b0000000000000011; //3
	   Whh[1][8] <= 16'b0000000000011100; //28
	   Whh[2][0] <= 16'b0000000000000011; //3
	   Whh[2][1] <= 16'b1111111111111000; //-8
	   Whh[2][2] <= 16'b1111111111011101; //-35
	   Whh[2][3] <= 16'b0000000000000111; //7
	   Whh[2][4] <= 16'b0000000000011110; //30
	   Whh[2][5] <= 16'b0000000000010010; //18
	   Whh[2][6] <= 16'b1111111111100111; //-25
	   Whh[2][7] <= 16'b1111111111111010; //-6
	   Whh[2][8] <= 16'b1111111111011011; //-37
	   Whh[3][0] <= 16'b0000000000010010; //18
	   Whh[3][1] <= 16'b0000000000011100; //28
	   Whh[3][2] <= 16'b0000000000011110; //30
	   Whh[3][3] <= 16'b1111111111110000; //-16
	   Whh[3][4] <= 16'b0000000000101101; //45
	   Whh[3][5] <= 16'b0000000000010111; //23
	   Whh[3][6] <= 16'b0000000000001011; //11
	   Whh[3][7] <= 16'b0000000000000101; //5
	   Whh[3][8] <= 16'b1111111111110100; //-12
	   Whh[4][0] <= 16'b0000000000101011; //43
	   Whh[4][1] <= 16'b1111111111111111; //-1
	   Whh[4][2] <= 16'b1111111111010110; //-42
	   Whh[4][3] <= 16'b1111111111110000; //-16
	   Whh[4][4] <= 16'b0000000000000100; //4
	   Whh[4][5] <= 16'b0000000000011011; //27
	   Whh[4][6] <= 16'b1111111111101101; //-19
	   Whh[4][7] <= 16'b1111111111101110; //-18
	   Whh[4][8] <= 16'b0000000000101101; //45
	   Whh[5][0] <= 16'b0000000000001010; //10
	   Whh[5][1] <= 16'b0000000000011001; //25
	   Whh[5][2] <= 16'b1111111111101011; //-21
	   Whh[5][3] <= 16'b0000000000011111; //31
	   Whh[5][4] <= 16'b1111111111100101; //-27
	   Whh[5][5] <= 16'b1111111111011111; //-33
	   Whh[5][6] <= 16'b0000000000011100; //28
	   Whh[5][7] <= 16'b1111111111001011; //-53
	   Whh[5][8] <= 16'b0000000000111010; //58
	   Whh[6][0] <= 16'b1111111111111000; //-8
	   Whh[6][1] <= 16'b1111111111111010; //-6
	   Whh[6][2] <= 16'b0000000000001100; //12
	   Whh[6][3] <= 16'b1111111111110100; //-12
	   Whh[6][4] <= 16'b0000000000011000; //24
	   Whh[6][5] <= 16'b0000000000011111; //31
	   Whh[6][6] <= 16'b1111111111110000; //-16
	   Whh[6][7] <= 16'b1111111111011101; //-35
	   Whh[6][8] <= 16'b0000000000000001; //1
	   Whh[7][0] <= 16'b0000000000001110; //14
	   Whh[7][1] <= 16'b0000000000001001; //9
	   Whh[7][2] <= 16'b1111111111101000; //-24
	   Whh[7][3] <= 16'b0000000000101001; //41
	   Whh[7][4] <= 16'b1111111111111011; //-5
	   Whh[7][5] <= 16'b0000000000100001; //33
	   Whh[7][6] <= 16'b0000000000101010; //42
	   Whh[7][7] <= 16'b0000000000011010; //26
	   Whh[7][8] <= 16'b1111111111101011; //-21
	   Whh[8][0] <= 16'b1111111111101101; //-19
	   Whh[8][1] <= 16'b1111111111110100; //-12
	   Whh[8][2] <= 16'b0000000000010110; //22
	   Whh[8][3] <= 16'b1111111111101100; //-20
	   Whh[8][4] <= 16'b0000000000001001; //9
	   Whh[8][5] <= 16'b1111111111011010; //-38
	   Whh[8][6] <= 16'b1111111111110111; //-9
	   Whh[8][7] <= 16'b1111111111001001; //-55
	   Whh[8][8] <= 16'b0000000000100001; //33

	   Who[0] <= 16'b1111111111001010; //-54
	   Who[1] <= 16'b0000000000001010; //10
	   Who[2] <= 16'b1111111111111111; //-1
	   Who[3] <= 16'b0000000001010100; //84
	   Who[4] <= 16'b0000000000011110; //30
	   Who[5] <= 16'b0000000001000110; //70
	   Who[6] <= 16'b0000000001001110; //78
	   Who[7] <= 16'b0000000000101101; //45
	   Who[8] <= 16'b0000000000010010; //18

	   bih[0] <= 16'b0000000000001001; //9
	   bih[1] <= 16'b1111111111110110; //-10
	   bih[2] <= 16'b0000000000011101; //29
	   bih[3] <= 16'b1111111111111010; //-6
	   bih[4] <= 16'b0000000000001001; //9
	   bih[5] <= 16'b0000000000011100; //28
	   bih[6] <= 16'b0000000000001010; //10
	   bih[7] <= 16'b0000000000000111; //7
	   bih[8] <= 16'b0000000000000010; //2

	   bhh[0] <= 16'b0000000000100111; //39
	   bhh[1] <= 16'b0000000000000000; //0
	   bhh[2] <= 16'b1111111111010111; //-41
	   bhh[3] <= 16'b0000000000000110; //6
	   bhh[4] <= 16'b1111111111111101; //-3
	   bhh[5] <= 16'b1111111111111000; //-8
	   bhh[6] <= 16'b0000000000001101; //13
	   bhh[7] <= 16'b1111111111100010; //-30
	   bhh[8] <= 16'b1111111111101110; //-18

	   bho <= 16'b0000000000000000; //0

		for (i=0; i<N_HIDDEN; i=i+1) begin
			WiXi[i] <= (Wih[i]*data)>>>SFT_R;
			WiXibi[i] <= WiXi[i] + bih[i];
		end

		for (i=0; i<N_HIDDEN; i=i+1) begin
			WhHi0[i] <= (Whh[0][i]*WhHibh_prev[i])>>>SFT_R; //PRECISION; 
			WhHi1[i] <= (Whh[1][i]*WhHibh_prev[i])>>>SFT_R; //PRECISION; 
			WhHi2[i] <= (Whh[2][i]*WhHibh_prev[i])>>>SFT_R; //PRECISION; 
			WhHi3[i] <= (Whh[3][i]*WhHibh_prev[i])>>>SFT_R; //PRECISION; 
			WhHi4[i] <= (Whh[4][i]*WhHibh_prev[i])>>>SFT_R; //PRECISION; 
			WhHi5[i] <= (Whh[5][i]*WhHibh_prev[i])>>>SFT_R; //PRECISION; 
			WhHi6[i] <= (Whh[6][i]*WhHibh_prev[i])>>>SFT_R; //PRECISION; 
			WhHi7[i] <= (Whh[7][i]*WhHibh_prev[i])>>>SFT_R; //PRECISION; 
			WhHi8[i] <= (Whh[8][i]*WhHibh_prev[i])>>>SFT_R; //PRECISION; 
		end
			
		WhHi0_SUM <= WhHi0[0]+WhHi0[1]+WhHi0[2]+WhHi0[3]+WhHi0[4]+WhHi0[5]+WhHi0[6]+WhHi0[7]+WhHi0[8]+bhh[0];
		WhHi1_SUM <= WhHi1[0]+WhHi1[1]+WhHi1[2]+WhHi1[3]+WhHi1[4]+WhHi1[5]+WhHi1[6]+WhHi1[7]+WhHi1[8]+bhh[1];
		WhHi2_SUM <= WhHi2[0]+WhHi2[1]+WhHi2[2]+WhHi2[3]+WhHi2[4]+WhHi2[5]+WhHi2[6]+WhHi2[7]+WhHi2[8]+bhh[2];
		WhHi3_SUM <= WhHi3[0]+WhHi3[1]+WhHi3[2]+WhHi3[3]+WhHi3[4]+WhHi3[5]+WhHi3[6]+WhHi3[7]+WhHi3[8]+bhh[3];
		WhHi4_SUM <= WhHi4[0]+WhHi4[1]+WhHi4[2]+WhHi4[3]+WhHi4[4]+WhHi4[5]+WhHi4[6]+WhHi4[7]+WhHi4[8]+bhh[4];
		WhHi5_SUM <= WhHi5[0]+WhHi5[1]+WhHi5[2]+WhHi5[3]+WhHi5[4]+WhHi5[5]+WhHi5[6]+WhHi5[7]+WhHi5[8]+bhh[5];
		WhHi6_SUM <= WhHi6[0]+WhHi6[1]+WhHi6[2]+WhHi6[3]+WhHi6[4]+WhHi6[5]+WhHi6[6]+WhHi6[7]+WhHi6[8]+bhh[6];
		WhHi7_SUM <= WhHi7[0]+WhHi7[1]+WhHi7[2]+WhHi7[3]+WhHi7[4]+WhHi7[5]+WhHi7[6]+WhHi7[7]+WhHi7[8]+bhh[7];
		WhHi8_SUM <= WhHi8[0]+WhHi8[1]+WhHi8[2]+WhHi8[3]+WhHi8[4]+WhHi8[5]+WhHi8[6]+WhHi8[7]+WhHi8[8]+bhh[8];

		if (!Hi_init) begin
			WhHibh[0] <= bhh[0];
			WhHibh[1] <= bhh[1];
			WhHibh[2] <= bhh[2];
			WhHibh[3] <= bhh[3];
			WhHibh[4] <= bhh[4];
			WhHibh[5] <= bhh[5];
			WhHibh[6] <= bhh[6];
			WhHibh[7] <= bhh[7];
			WhHibh[8] <= bhh[8];
		end
		else begin
			WhHibh[0] <= WhHi0_SUM;
			WhHibh[1] <= WhHi1_SUM;
			WhHibh[2] <= WhHi2_SUM;
			WhHibh[3] <= WhHi3_SUM;
			WhHibh[4] <= WhHi4_SUM;
			WhHibh[5] <= WhHi5_SUM;
			WhHibh[6] <= WhHi6_SUM;
			WhHibh[7] <= WhHi7_SUM;
			WhHibh[8] <= WhHi8_SUM;
		end

		yhat0_t <= (Who[0]*Ho0)>>>SFT_R;
		yhat1_t <= (Who[1]*Ho1)>>>SFT_R;
		yhat2_t <= (Who[2]*Ho2)>>>SFT_R;
		yhat3_t <= (Who[3]*Ho3)>>>SFT_R;
		yhat4_t <= (Who[4]*Ho4)>>>SFT_R;
		yhat5_t <= (Who[5]*Ho5)>>>SFT_R;
		yhat6_t <= (Who[6]*Ho6)>>>SFT_R;
		yhat7_t <= (Who[7]*Ho7)>>>SFT_R;
		yhat8_t <= (Who[8]*Ho8)>>>SFT_R;
		yhat <= yhat0_t + yhat1_t + yhat2_t + yhat3_t + yhat4_t + yhat5_t + yhat6_t + yhat7_t + yhat8_t + bho;
	end

//--------------------------------------------------------------------------
//		Inserting FFs for timing closure of computation blocks
//--------------------------------------------------------------------------
	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			for (i=0; i<N_HIDDEN; i=i+1) WhHibh_d[i] <= 0;
			for (i=0; i<N_HIDDEN; i=i+1) WiXibi_d[i] <= 0;
			yhat_d <= 0;
 			Ho0 <= 0;
 			Ho1 <= 0;
 			Ho2 <= 0;
 			Ho3 <= 0;
 			Ho4 <= 0;
 			Ho5 <= 0;
 			Ho6 <= 0;
 			Ho7 <= 0;
 			Ho8 <= 0;
		end
		else begin 
			if (matmult_done) begin
				for (i=0; i<N_HIDDEN; i=i+1) WhHibh_d[i] <= WhHibh[i];
				for (i=0; i<N_HIDDEN; i=i+1) WiXibi_d[i] <= WiXibi[i];
			end
			if (pred_done) yhat_d <= yhat;
			if (tanh_done) begin
 				Ho0 <= Ho0_t;
 				Ho1 <= Ho1_t;
 				Ho2 <= Ho2_t;
 				Ho3 <= Ho3_t;
 				Ho4 <= Ho4_t;
 				Ho5 <= Ho5_t;
 				Ho6 <= Ho6_t;
 				Ho7 <= Ho7_t;
 				Ho8 <= Ho8_t;
			end
 		end
	end

	assign output_mem = output_mem_d;

	assign matmult_done_t = (scheduler == 3) ? 1 : 0;
	assign tanh_done_t = (scheduler == 4) ? 1 : 0;
	assign pred_done_t = (scheduler == 6) ? 1 : 0;

	assign Hi0 = WiXibi_out0 + WhHibh_out0;
	assign Hi1 = WiXibi_out1 + WhHibh_out1;
	assign Hi2 = WiXibi_out2 + WhHibh_out2;
	assign Hi3 = WiXibi_out3 + WhHibh_out3;
	assign Hi4 = WiXibi_out4 + WhHibh_out4;
	assign Hi5 = WiXibi_out5 + WhHibh_out5;
	assign Hi6 = WiXibi_out6 + WhHibh_out6;
	assign Hi7 = WiXibi_out7 + WhHibh_out7;
	assign Hi8 = WiXibi_out8 + WhHibh_out8;

	assign data = sram_data_in;
	assign WiXibi_out0_t = WiXibi_d[0];
	assign WiXibi_out1_t = WiXibi_d[1];
	assign WiXibi_out2_t = WiXibi_d[2];	
	assign WiXibi_out3_t = WiXibi_d[3];
	assign WiXibi_out4_t = WiXibi_d[4];
	assign WiXibi_out5_t = WiXibi_d[5];
	assign WiXibi_out6_t = WiXibi_d[6];	
	assign WiXibi_out7_t = WiXibi_d[7];
	assign WiXibi_out8_t = WiXibi_d[8];
	assign WhHibh_out0_t = WhHibh_d[0];
	assign WhHibh_out1_t = WhHibh_d[1];
	assign WhHibh_out2_t = WhHibh_d[2];	
	assign WhHibh_out3_t = WhHibh_d[3];
	assign WhHibh_out4_t = WhHibh_d[4];
	assign WhHibh_out5_t = WhHibh_d[5];
	assign WhHibh_out6_t = WhHibh_d[6];	
	assign WhHibh_out7_t = WhHibh_d[7];
	assign WhHibh_out8_t = WhHibh_d[8];

	assign WiXibi_out0 = WiXibi_out0_t[DATA_WIDTH-1:0];
	assign WiXibi_out1 = WiXibi_out1_t[DATA_WIDTH-1:0];
	assign WiXibi_out2 = WiXibi_out2_t[DATA_WIDTH-1:0];	
	assign WiXibi_out3 = WiXibi_out3_t[DATA_WIDTH-1:0];
	assign WiXibi_out4 = WiXibi_out4_t[DATA_WIDTH-1:0];
	assign WiXibi_out5 = WiXibi_out5_t[DATA_WIDTH-1:0];
	assign WiXibi_out6 = WiXibi_out6_t[DATA_WIDTH-1:0];	
	assign WiXibi_out7 = WiXibi_out7_t[DATA_WIDTH-1:0];
	assign WiXibi_out8 = WiXibi_out8_t[DATA_WIDTH-1:0];
	assign WhHibh_out0 = WhHibh_out0_t[DATA_WIDTH-1:0];
	assign WhHibh_out1 = WhHibh_out1_t[DATA_WIDTH-1:0];
	assign WhHibh_out2 = WhHibh_out2_t[DATA_WIDTH-1:0];	
	assign WhHibh_out3 = WhHibh_out3_t[DATA_WIDTH-1:0];
	assign WhHibh_out4 = WhHibh_out4_t[DATA_WIDTH-1:0];
	assign WhHibh_out5 = WhHibh_out5_t[DATA_WIDTH-1:0];
	assign WhHibh_out6 = WhHibh_out6_t[DATA_WIDTH-1:0];	
	assign WhHibh_out7 = WhHibh_out7_t[DATA_WIDTH-1:0];
	assign WhHibh_out8 = WhHibh_out8_t[DATA_WIDTH-1:0];
endmodule

//--------------------------------------------------------------------------
//		Y = tanh(X) : linear approximation within dynamic ranges
//--------------------------------------------------------------------------
module tanh(X,Y);
	parameter DATA_WIDTH = 16;
	parameter NEGPRE = -384;
	parameter POSPRE = 384;
	
	input signed [DATA_WIDTH-1:0] X;
	output wire signed [DATA_WIDTH-1:0] Y;

	assign Y = (X[DATA_WIDTH-1]) ? ( (X < NEGPRE)? -128 : X ) : ( (X > POSPRE) ? 128 : X );
endmodule


