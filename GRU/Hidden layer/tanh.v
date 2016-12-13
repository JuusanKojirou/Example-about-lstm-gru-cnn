`timescale 1ns / 1ns
module tanh(
	input clk,
	input rst_n,
	input en,
	input signed [15:0] tanh_input,
	output wire result_valid,
	output wire signed [15:0] tanh_output
    );
    
//TAKE 4 CLK CYCLES

	wire signed [15:0] result;
	wire signed [15:0] inl1;

	assign inl1 = tanh_input<<<1;
    sigmoid SIG(
    	.clk(clk),
    	.rst_n(rst_n),
    	.en(en),
    	.sigmoid_input(inl1),
    	.result_valid(result_valid),
    	.sigmoid_output(result));
	
	// always @(posedge clk or negedge rst_n)begin
	// 	if(~rst_n) begin
	// 		tanh_output <= 0;
	// 	end
	// 	else begin
	// 		tanh_output <= (result<<<1) - 16'b0100000000000000;
	// 	end
	// end

	assign tanh_output = (result<<<1) - 16'b0100000000000000;

endmodule
