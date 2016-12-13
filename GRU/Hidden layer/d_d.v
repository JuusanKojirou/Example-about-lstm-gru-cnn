//`include "parameter"

module d_d(
	input clk, 
	input rst_n, 
	input en,
	input signed [DATABIT-1:0] rorz, 
	input [HTNUM-1:0] w,
	input signed [DATABIT-1:0] dh0_dw,
	input signed [DATABIT-1:0] dh1_dw,
	input signed [DATABIT-1:0] dh2_dw,
	input signed [DATABIT-1:0] dh3_dw,
	output result_valid,
	output signed [DATABIT-1:0] out
	);

//take 6 clk cycles

wire signed [DATABIT-1:0] t1;
wire signed [DATABIT-1:0] ff;
wire signed [DATABIT-1:0] t2;

//BIT
assign t1 = ~rorz + 16'b0100000000000001;

//BIT,CELL

wire result_valid0;
wire result_valid1;

mult16x16_2int M1(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(rorz),
	.multiplicator2(t1),
	.result_valid(result_valid0),
	.result(t2)
	);

vector_mult_4_parallel VM1(
	.clk(clk), 
	.rst_n(rst_n), 
	.en(en), 
	.a_0(dh0_dw), 
	.a_1(dh1_dw), 
	.a_2(dh2_dw), 
	.a_3(dh3_dw), 
	.b_0(w[15:0]), 
	.b_1(w[31:16]), 
	.b_2(w[47:32]), 
	.b_3(w[63:48]), 
	.result_valid(result_valid1),
	.ab_out(ff)
	);

mult16x16_2int M2(
	.clk(clk),
	.rst_n(rst_n),
	.en(result_valid0&result_valid1),
	.multiplicator1(ff),
	.multiplicator2(t2),
	.result_valid(result_valid),
	.result(out)
	);

// always @(*) begin
// 	out = t3;
// 	end

endmodule

