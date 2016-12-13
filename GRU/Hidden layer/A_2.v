//`include "parameter.v"
module A_2(
	input clk,
	input rst_n,
	input en,
	input signed [DATABIT-1:0] xt,
	input [DATABIT-1:0] htb,
	input [DATABIT-1:0] zt,
	output wire result_valid,
	output wire signed [DATABIT-1:0] result
);

wire signed [DATABIT-1:0] htbd;
wire rv1;
mult16x16_2int htb2(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(htb[15:0]),
	.multiplicator2(htb[15:0]),
	.result_valid(rv1),
	.result(htbd)
);
wire signed [DATABIT-1:0] onesubhtbd;
assign onesubhtbd = ~htbd+ 16'b0100000000000001;

wire signed [DATABIT-1:0] onesubzt;
assign onesubzt = ~zt+ 16'b0100000000000001;

wire signed [DATABIT-1:0] hhx;
wire rv2;
mult16x16_2int Mh0(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(xt),
	.multiplicator2(onesubzt),
	.result_valid(rv2),
	.result(hhx)
	);


mult16x16_2int fa(
	.clk(clk),
	.rst_n(rst_n),
	.en(rv1&rv2),
	.multiplicator1(onesubhtbd),
	.multiplicator2(hhx),
	.result_valid(result_valid),
	.result(result)
	);


endmodule