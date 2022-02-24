`timescale 1ns/1ps
module tb_rnn_comp();
	
	// Testbench variables
	parameter DATA_WIDTH = 16;			// Input data width for 2's complement numbers w/ variable precision
	parameter PRECISION = 128;			// Optimum precision of the RNN algorithm
	parameter SFT_R = 7;					// Decimal point handling to use signed variables

	reg clk = 0; 							// Clock
	reg rst_n;								// Reset
	reg data_valid;						// Input data valid signal - input is fetched when it is 1
	reg [DATA_WIDTH-1:0] data_in;		// Input data
	reg read_start;						// All input data are buffered and ready to compute
	reg int_clear;							// Interrupt clear signal - for future use for interrupt handling
	reg [8:0] read_address;				// Read address for predicted values of RNN
	wire complete;							// Complete the computation and ready to read the final results
	wire signed [DATA_WIDTH*2-1:0] output_mem;	// Prediction results
	
	// Instantiate the DUT
	rnn_comp #(.DATA_WIDTH(DATA_WIDTH), .PRECISION(PRECISION), .SFT_R(SFT_R)) RNN_COMP
	(
		.clk					(clk), 
		.rst_n				(rst_n), 
		.data_valid			(data_valid),  
		.data_in				(data_in), 
		.read_start			(read_start), 
		.int_clear			(int_clear),  
		.read_address		(read_address),
		.complete			(complete),
		.output_mem			(output_mem)
	);

	integer i, f;
	reg signed [DATA_WIDTH*2-1:0] out [499:0];

//----------------------------------------------------------------
//   To write input data into input memory  
//   		- data_in: 3 clocks
//   		- data_valid: 010 on 3 clocks 
//----------------------------------------------------------------
	task write_data(input [DATA_WIDTH-1:0] d_in); 
		begin
			@(posedge clk); // sync to positive edge of clock
				data_valid = 0;
				data_in = d_in;
			@(posedge clk); // sync to positive edge of clock
				data_valid = 1;
				data_in = d_in;
			@(posedge clk); // sync to positive edge of clock
				data_valid = 0;
				data_in = d_in;
		end
	endtask

//----------------------------------------------------------------
//   To read the predicted results from output memory
//   		- read_address: 2 clocks
//----------------------------------------------------------------
	task read_addr(input [8:0] addr); 
		begin
			@(posedge clk); // sync to positive edge of clock
				read_address = addr;
			@(posedge clk); // sync to positive edge of clock
				read_address = addr;
		end
	endtask

//----------------------------------------------------------------
//   To start computation
//   		ex) to generate 1 clock pulse -> enable(1); enable(0);
//----------------------------------------------------------------
	task enable(input en); 
		begin
			@(posedge clk); // sync to positive edge of clock
				read_start = en;
		end
	endtask

//----------------------------------------------------------------
//   To clear interrupt (for future use)
//   		ex) to generate 1 clock pulse -> clear(1); clear(0);
//----------------------------------------------------------------
	task clear(input clear); 
		begin
			@(posedge clk); // sync to positive edge of clock
				int_clear = clear;
		end
	endtask

	// Create the clock signal
	always begin #0.5 clk = ~clk; end

    // Create stimulus	  
	initial begin

	#1; 
	rst_n = 0; data_valid = 0; read_start = 0; int_clear = 0; // Initialization
		
	#1.3; 
	rst_n = 1;

   write_data(16'b0000000001000101);  //69
   write_data(16'b0000000000111000);  //56
   write_data(16'b0000000000101100);  //44
   write_data(16'b0000000000100001);  //33
   write_data(16'b0000000000011000);  //24
   write_data(16'b0000000000010001);  //17
   write_data(16'b0000000000001011);  //11
   write_data(16'b0000000000000111);  //7
   write_data(16'b0000000000000100);  //4
   write_data(16'b0000000000000010);  //2
   write_data(16'b0000000000000001);  //1
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b1111111111111111);  //-1
   write_data(16'b1111111111111110);  //-2
   write_data(16'b1111111111111100);  //-4
   write_data(16'b1111111111111001);  //-7
   write_data(16'b1111111111110101);  //-11
   write_data(16'b1111111111101111);  //-17
   write_data(16'b1111111111100111);  //-25
   write_data(16'b1111111111011110);  //-34
   write_data(16'b1111111111010011);  //-45
   write_data(16'b1111111111000111);  //-57
   write_data(16'b1111111110111010);  //-70
   write_data(16'b1111111110101100);  //-84
   write_data(16'b1111111110011111);  //-97
   write_data(16'b1111111110010010);  //-110
   write_data(16'b1111111110001000);  //-120
   write_data(16'b1111111110000010);  //-126
   write_data(16'b1111111110000000);  //-128
   write_data(16'b1111111110000100);  //-124
   write_data(16'b1111111110001111);  //-113
   write_data(16'b1111111110100000);  //-96
   write_data(16'b1111111110110111);  //-73
   write_data(16'b1111111111010011);  //-45
   write_data(16'b1111111111110010);  //-14
   write_data(16'b0000000000010010);  //18
   write_data(16'b0000000000110001);  //49
   write_data(16'b0000000001001101);  //77
   write_data(16'b0000000001100011);  //99
   write_data(16'b0000000001110011);  //115
   write_data(16'b0000000001111101);  //125
   write_data(16'b0000000010000000);  //128
   write_data(16'b0000000001111101);  //125
   write_data(16'b0000000001110110);  //118
   write_data(16'b0000000001101100);  //108
   write_data(16'b0000000001011111);  //95
   write_data(16'b0000000001010010);  //82
   write_data(16'b0000000001000100);  //68
   write_data(16'b0000000000110111);  //55
   write_data(16'b0000000000101011);  //43
   write_data(16'b0000000000100000);  //32
   write_data(16'b0000000000011000);  //24
   write_data(16'b0000000000010000);  //16
   write_data(16'b0000000000001011);  //11
   write_data(16'b0000000000000110);  //6
   write_data(16'b0000000000000100);  //4
   write_data(16'b0000000000000010);  //2
   write_data(16'b0000000000000001);  //1
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b1111111111111111);  //-1
   write_data(16'b1111111111111110);  //-2
   write_data(16'b1111111111111100);  //-4
   write_data(16'b1111111111111001);  //-7
   write_data(16'b1111111111110100);  //-12
   write_data(16'b1111111111101110);  //-18
   write_data(16'b1111111111100110);  //-26
   write_data(16'b1111111111011101);  //-35
   write_data(16'b1111111111010010);  //-46
   write_data(16'b1111111111000110);  //-58
   write_data(16'b1111111110111001);  //-71
   write_data(16'b1111111110101011);  //-85
   write_data(16'b1111111110011110);  //-98
   write_data(16'b1111111110010001);  //-111
   write_data(16'b1111111110000111);  //-121
   write_data(16'b1111111110000001);  //-127
   write_data(16'b1111111110000000);  //-128
   write_data(16'b1111111110000101);  //-123
   write_data(16'b1111111110010000);  //-112
   write_data(16'b1111111110100010);  //-94
   write_data(16'b1111111110111010);  //-70
   write_data(16'b1111111111010110);  //-42
   write_data(16'b1111111111110110);  //-10
   write_data(16'b0000000000010110);  //22
   write_data(16'b0000000000110100);  //52
   write_data(16'b0000000001001111);  //79
   write_data(16'b0000000001100101);  //101
   write_data(16'b0000000001110101);  //117
   write_data(16'b0000000001111101);  //125
   write_data(16'b0000000010000000);  //128
   write_data(16'b0000000001111101);  //125
   write_data(16'b0000000001110101);  //117
   write_data(16'b0000000001101011);  //107
   write_data(16'b0000000001011110);  //94
   write_data(16'b0000000001010000);  //80
   write_data(16'b0000000001000010);  //66
   write_data(16'b0000000000110101);  //53
   write_data(16'b0000000000101010);  //42
   write_data(16'b0000000000011111);  //31
   write_data(16'b0000000000010111);  //23
   write_data(16'b0000000000010000);  //16
   write_data(16'b0000000000001010);  //10
   write_data(16'b0000000000000110);  //6
   write_data(16'b0000000000000011);  //3
   write_data(16'b0000000000000001);  //1
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b1111111111111111);  //-1
   write_data(16'b1111111111111110);  //-2
   write_data(16'b1111111111111100);  //-4
   write_data(16'b1111111111111000);  //-8
   write_data(16'b1111111111110100);  //-12
   write_data(16'b1111111111101101);  //-19
   write_data(16'b1111111111100110);  //-26
   write_data(16'b1111111111011100);  //-36
   write_data(16'b1111111111010001);  //-47
   write_data(16'b1111111111000101);  //-59
   write_data(16'b1111111110110111);  //-73
   write_data(16'b1111111110101010);  //-86
   write_data(16'b1111111110011100);  //-100
   write_data(16'b1111111110010000);  //-112
   write_data(16'b1111111110000111);  //-121
   write_data(16'b1111111110000001);  //-127
   write_data(16'b1111111110000000);  //-128
   write_data(16'b1111111110000110);  //-122
   write_data(16'b1111111110010010);  //-110
   write_data(16'b1111111110100100);  //-92
   write_data(16'b1111111110111100);  //-68
   write_data(16'b1111111111011001);  //-39
   write_data(16'b1111111111111001);  //-7
   write_data(16'b0000000000011001);  //25
   write_data(16'b0000000000110111);  //55
   write_data(16'b0000000001010010);  //82
   write_data(16'b0000000001100111);  //103
   write_data(16'b0000000001110110);  //118
   write_data(16'b0000000001111110);  //126
   write_data(16'b0000000010000000);  //128
   write_data(16'b0000000001111100);  //124
   write_data(16'b0000000001110100);  //116
   write_data(16'b0000000001101001);  //105
   write_data(16'b0000000001011101);  //93
   write_data(16'b0000000001001111);  //79
   write_data(16'b0000000001000001);  //65
   write_data(16'b0000000000110100);  //52
   write_data(16'b0000000000101001);  //41
   write_data(16'b0000000000011110);  //30
   write_data(16'b0000000000010110);  //22
   write_data(16'b0000000000001111);  //15
   write_data(16'b0000000000001010);  //10
   write_data(16'b0000000000000110);  //6
   write_data(16'b0000000000000011);  //3
   write_data(16'b0000000000000001);  //1
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b1111111111111111);  //-1
   write_data(16'b1111111111111110);  //-2
   write_data(16'b1111111111111011);  //-5
   write_data(16'b1111111111111000);  //-8
   write_data(16'b1111111111110011);  //-13
   write_data(16'b1111111111101101);  //-19
   write_data(16'b1111111111100101);  //-27
   write_data(16'b1111111111011011);  //-37
   write_data(16'b1111111111010000);  //-48
   write_data(16'b1111111111000100);  //-60
   write_data(16'b1111111110110110);  //-74
   write_data(16'b1111111110101000);  //-88
   write_data(16'b1111111110011011);  //-101
   write_data(16'b1111111110001111);  //-113
   write_data(16'b1111111110000110);  //-122
   write_data(16'b1111111110000001);  //-127
   write_data(16'b1111111110000001);  //-127
   write_data(16'b1111111110000111);  //-121
   write_data(16'b1111111110010011);  //-109
   write_data(16'b1111111110100110);  //-90
   write_data(16'b1111111110111111);  //-65
   write_data(16'b1111111111011100);  //-36
   write_data(16'b1111111111111100);  //-4
   write_data(16'b0000000000011100);  //28
   write_data(16'b0000000000111010);  //58
   write_data(16'b0000000001010100);  //84
   write_data(16'b0000000001101001);  //105
   write_data(16'b0000000001110111);  //119
   write_data(16'b0000000001111110);  //126
   write_data(16'b0000000010000000);  //128
   write_data(16'b0000000001111100);  //124
   write_data(16'b0000000001110011);  //115
   write_data(16'b0000000001101000);  //104
   write_data(16'b0000000001011011);  //91
   write_data(16'b0000000001001101);  //77
   write_data(16'b0000000001000000);  //64
   write_data(16'b0000000000110011);  //51
   write_data(16'b0000000000101000);  //40
   write_data(16'b0000000000011110);  //30
   write_data(16'b0000000000010101);  //21
   write_data(16'b0000000000001110);  //14
   write_data(16'b0000000000001001);  //9
   write_data(16'b0000000000000101);  //5
   write_data(16'b0000000000000011);  //3
   write_data(16'b0000000000000001);  //1
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b1111111111111111);  //-1
   write_data(16'b1111111111111101);  //-3
   write_data(16'b1111111111111011);  //-5
   write_data(16'b1111111111110111);  //-9
   write_data(16'b1111111111110010);  //-14
   write_data(16'b1111111111101100);  //-20
   write_data(16'b1111111111100100);  //-28
   write_data(16'b1111111111011010);  //-38
   write_data(16'b1111111111001111);  //-49
   write_data(16'b1111111111000010);  //-62
   write_data(16'b1111111110110101);  //-75
   write_data(16'b1111111110100111);  //-89
   write_data(16'b1111111110011010);  //-102
   write_data(16'b1111111110001110);  //-114
   write_data(16'b1111111110000101);  //-123
   write_data(16'b1111111110000000);  //-128
   write_data(16'b1111111110000001);  //-127
   write_data(16'b1111111110001000);  //-120
   write_data(16'b1111111110010101);  //-107
   write_data(16'b1111111110101001);  //-87
   write_data(16'b1111111111000010);  //-62
   write_data(16'b1111111111011111);  //-33
   write_data(16'b1111111111111111);  //-1
   write_data(16'b0000000000011111);  //31
   write_data(16'b0000000000111101);  //61
   write_data(16'b0000000001010110);  //86
   write_data(16'b0000000001101010);  //106
   write_data(16'b0000000001111000);  //120
   write_data(16'b0000000001111111);  //127
   write_data(16'b0000000010000000);  //128
   write_data(16'b0000000001111011);  //123
   write_data(16'b0000000001110010);  //114
   write_data(16'b0000000001100111);  //103
   write_data(16'b0000000001011010);  //90
   write_data(16'b0000000001001100);  //76
   write_data(16'b0000000000111110);  //62
   write_data(16'b0000000000110010);  //50
   write_data(16'b0000000000100110);  //38
   write_data(16'b0000000000011101);  //29
   write_data(16'b0000000000010100);  //20
   write_data(16'b0000000000001110);  //14
   write_data(16'b0000000000001001);  //9
   write_data(16'b0000000000000101);  //5
   write_data(16'b0000000000000011);  //3
   write_data(16'b0000000000000001);  //1
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b1111111111111111);  //-1
   write_data(16'b1111111111111101);  //-3
   write_data(16'b1111111111111011);  //-5
   write_data(16'b1111111111110111);  //-9
   write_data(16'b1111111111110010);  //-14
   write_data(16'b1111111111101011);  //-21
   write_data(16'b1111111111100011);  //-29
   write_data(16'b1111111111011001);  //-39
   write_data(16'b1111111111001110);  //-50
   write_data(16'b1111111111000001);  //-63
   write_data(16'b1111111110110011);  //-77
   write_data(16'b1111111110100110);  //-90
   write_data(16'b1111111110011000);  //-104
   write_data(16'b1111111110001101);  //-115
   write_data(16'b1111111110000101);  //-123
   write_data(16'b1111111110000000);  //-128
   write_data(16'b1111111110000001);  //-127
   write_data(16'b1111111110001001);  //-119
   write_data(16'b1111111110010111);  //-105
   write_data(16'b1111111110101011);  //-85
   write_data(16'b1111111111000101);  //-59
   write_data(16'b1111111111100010);  //-30
   write_data(16'b0000000000000010);  //2
   write_data(16'b0000000000100010);  //34
   write_data(16'b0000000001000000);  //64
   write_data(16'b0000000001011001);  //89
   write_data(16'b0000000001101100);  //108
   write_data(16'b0000000001111001);  //121
   write_data(16'b0000000001111111);  //127
   write_data(16'b0000000001111111);  //127
   write_data(16'b0000000001111010);  //122
   write_data(16'b0000000001110001);  //113
   write_data(16'b0000000001100110);  //102
   write_data(16'b0000000001011000);  //88
   write_data(16'b0000000001001011);  //75
   write_data(16'b0000000000111101);  //61
   write_data(16'b0000000000110001);  //49
   write_data(16'b0000000000100101);  //37
   write_data(16'b0000000000011100);  //28
   write_data(16'b0000000000010100);  //20
   write_data(16'b0000000000001101);  //13
   write_data(16'b0000000000001000);  //8
   write_data(16'b0000000000000101);  //5
   write_data(16'b0000000000000010);  //2
   write_data(16'b0000000000000001);  //1
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b1111111111111111);  //-1
   write_data(16'b1111111111111101);  //-3
   write_data(16'b1111111111111010);  //-6
   write_data(16'b1111111111110111);  //-9
   write_data(16'b1111111111110001);  //-15
   write_data(16'b1111111111101010);  //-22
   write_data(16'b1111111111100010);  //-30
   write_data(16'b1111111111011000);  //-40
   write_data(16'b1111111111001100);  //-52
   write_data(16'b1111111111000000);  //-64
   write_data(16'b1111111110110010);  //-78
   write_data(16'b1111111110100100);  //-92
   write_data(16'b1111111110010111);  //-105
   write_data(16'b1111111110001100);  //-116
   write_data(16'b1111111110000100);  //-124
   write_data(16'b1111111110000000);  //-128
   write_data(16'b1111111110000010);  //-126
   write_data(16'b1111111110001010);  //-118
   write_data(16'b1111111110011000);  //-104
   write_data(16'b1111111110101101);  //-83
   write_data(16'b1111111111000111);  //-57
   write_data(16'b1111111111100110);  //-26
   write_data(16'b0000000000000110);  //6
   write_data(16'b0000000000100101);  //37
   write_data(16'b0000000001000010);  //66
   write_data(16'b0000000001011011);  //91
   write_data(16'b0000000001101110);  //110
   write_data(16'b0000000001111010);  //122
   write_data(16'b0000000001111111);  //127
   write_data(16'b0000000001111111);  //127
   write_data(16'b0000000001111010);  //122
   write_data(16'b0000000001110000);  //112
   write_data(16'b0000000001100100);  //100
   write_data(16'b0000000001010111);  //87
   write_data(16'b0000000001001001);  //73
   write_data(16'b0000000000111100);  //60
   write_data(16'b0000000000101111);  //47
   write_data(16'b0000000000100100);  //36
   write_data(16'b0000000000011011);  //27
   write_data(16'b0000000000010011);  //19
   write_data(16'b0000000000001101);  //13
   write_data(16'b0000000000001000);  //8
   write_data(16'b0000000000000101);  //5
   write_data(16'b0000000000000010);  //2
   write_data(16'b0000000000000001);  //1
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b1111111111111111);  //-1
   write_data(16'b1111111111111101);  //-3
   write_data(16'b1111111111111010);  //-6
   write_data(16'b1111111111110110);  //-10
   write_data(16'b1111111111110001);  //-15
   write_data(16'b1111111111101010);  //-22
   write_data(16'b1111111111100001);  //-31
   write_data(16'b1111111111010111);  //-41
   write_data(16'b1111111111001011);  //-53
   write_data(16'b1111111110111110);  //-66
   write_data(16'b1111111110110001);  //-79
   write_data(16'b1111111110100011);  //-93
   write_data(16'b1111111110010110);  //-106
   write_data(16'b1111111110001011);  //-117
   write_data(16'b1111111110000011);  //-125
   write_data(16'b1111111110000000);  //-128
   write_data(16'b1111111110000010);  //-126
   write_data(16'b1111111110001011);  //-117
   write_data(16'b1111111110011010);  //-102
   write_data(16'b1111111110110000);  //-80
   write_data(16'b1111111111001010);  //-54
   write_data(16'b1111111111101001);  //-23
   write_data(16'b0000000000001001);  //9
   write_data(16'b0000000000101000);  //40
   write_data(16'b0000000001000101);  //69
   write_data(16'b0000000001011101);  //93
   write_data(16'b0000000001101111);  //111
   write_data(16'b0000000001111011);  //123
   write_data(16'b0000000010000000);  //128
   write_data(16'b0000000001111111);  //127
   write_data(16'b0000000001111001);  //121
   write_data(16'b0000000001101111);  //111
   write_data(16'b0000000001100011);  //99
   write_data(16'b0000000001010110);  //86
   write_data(16'b0000000001001000);  //72
   write_data(16'b0000000000111011);  //59
   write_data(16'b0000000000101110);  //46
   write_data(16'b0000000000100011);  //35
   write_data(16'b0000000000011010);  //26
   write_data(16'b0000000000010010);  //18
   write_data(16'b0000000000001100);  //12
   write_data(16'b0000000000001000);  //8
   write_data(16'b0000000000000100);  //4
   write_data(16'b0000000000000010);  //2
   write_data(16'b0000000000000001);  //1
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b1111111111111111);  //-1
   write_data(16'b1111111111111110);  //-2
   write_data(16'b1111111111111101);  //-3
   write_data(16'b1111111111111010);  //-6
   write_data(16'b1111111111110110);  //-10
   write_data(16'b1111111111110000);  //-16
   write_data(16'b1111111111101001);  //-23
   write_data(16'b1111111111100000);  //-32
   write_data(16'b1111111111010110);  //-42
   write_data(16'b1111111111001010);  //-54
   write_data(16'b1111111110111101);  //-67
   write_data(16'b1111111110101111);  //-81
   write_data(16'b1111111110100001);  //-95
   write_data(16'b1111111110010101);  //-107
   write_data(16'b1111111110001010);  //-118
   write_data(16'b1111111110000011);  //-125
   write_data(16'b1111111110000000);  //-128
   write_data(16'b1111111110000011);  //-125
   write_data(16'b1111111110001100);  //-116
   write_data(16'b1111111110011100);  //-100
   write_data(16'b1111111110110010);  //-78
   write_data(16'b1111111111001101);  //-51
   write_data(16'b1111111111101100);  //-20
   write_data(16'b0000000000001100);  //12
   write_data(16'b0000000000101011);  //43
   write_data(16'b0000000001001000);  //72
   write_data(16'b0000000001011111);  //95
   write_data(16'b0000000001110001);  //113
   write_data(16'b0000000001111011);  //123
   write_data(16'b0000000010000000);  //128
   write_data(16'b0000000001111110);  //126
   write_data(16'b0000000001111000);  //120
   write_data(16'b0000000001101110);  //110
   write_data(16'b0000000001100010);  //98
   write_data(16'b0000000001010100);  //84
   write_data(16'b0000000001000111);  //71
   write_data(16'b0000000000111001);  //57
   write_data(16'b0000000000101101);  //45
   write_data(16'b0000000000100010);  //34
   write_data(16'b0000000000011001);  //25
   write_data(16'b0000000000010010);  //18
   write_data(16'b0000000000001100);  //12
   write_data(16'b0000000000000111);  //7
   write_data(16'b0000000000000100);  //4
   write_data(16'b0000000000000010);  //2
   write_data(16'b0000000000000001);  //1
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b0000000000000000);  //0
   write_data(16'b1111111111111111);  //-1
   write_data(16'b1111111111111110);  //-2
   write_data(16'b1111111111111100);  //-4
   write_data(16'b1111111111111001);  //-7
   write_data(16'b1111111111110101);  //-11
   write_data(16'b1111111111101111);  //-17
   write_data(16'b1111111111101000);  //-24
   write_data(16'b1111111111011111);  //-33
   write_data(16'b1111111111010101);  //-43
   write_data(16'b1111111111001001);  //-55
   write_data(16'b1111111110111100);  //-68
   write_data(16'b1111111110101110);  //-82
   write_data(16'b1111111110100000);  //-96
   write_data(16'b1111111110010100);  //-108
   write_data(16'b1111111110001001);  //-119
   write_data(16'b1111111110000010);  //-126
   write_data(16'b1111111110000000);  //-128
   write_data(16'b1111111110000011);  //-125
   write_data(16'b1111111110001101);  //-115
   write_data(16'b1111111110011110);  //-98
   write_data(16'b1111111110110101);  //-75
   write_data(16'b1111111111010000);  //-48
   write_data(16'b1111111111101111);  //-17
   write_data(16'b0000000000001111);  //15
   write_data(16'b0000000000101110);  //46
   write_data(16'b0000000001001010);  //74
   write_data(16'b0000000001100001);  //97
   write_data(16'b0000000001110010);  //114
   write_data(16'b0000000001111100);  //124
   write_data(16'b0000000010000000);  //128
   write_data(16'b0000000001111110);  //126
   write_data(16'b0000000001110111);  //119
   write_data(16'b0000000001101101);  //109
   write_data(16'b0000000001100001);  //97
   write_data(16'b0000000001010011);  //83
   write_data(16'b0000000001000101);  //69

	for (i=0;i<30;i=i+1) begin
		enable(0);
	end
	enable(1);
	enable(1);
	enable(0);
	#6300

	for (i=0;i<500;i=i+1) begin
   	read_addr(i); 
	end

	#1100
	clear(1);
	clear(0);

//	f = $fopen("output.txt","w");
	for (i=0;i<500;i=i+1) begin
		$display("%d, %d",i, out[i]);
//		$fwrite("%d, %d\n",i, out[i]);
	end
//	$fclose(f);

	#1100 
	$stop;

end
endmodule
