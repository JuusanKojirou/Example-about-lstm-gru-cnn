//`include "parameter.v"
module A_1(
	input clk,
	input rst_n,
	input en,
	input signed [DATABIT-1:0] xt,
	input signed [DATABIT-1:0] htb,
	input signed [DATABIT-1:0] ht1,
	input signed [DATABIT-1:0] zt,
	output wire result_valid,
	output signed [DATABIT-1:0] result
);

//take 6 clk cycles

wire signed [DATABIT-1:0] onesubzt;
assign onesubzt = ~zt+ 16'b0100000000000001;

wire signed [DATABIT-1:0] zt1zt;
wire rv_z;

mult16x16_2int M0(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(zt),
	.multiplicator2(onesubzt),
	.result_valid(rv_z),
	.result(zt1zt)
);



wire signed [DATABIT-1:0] ht1subhtb;
assign ht1subhtb = ht1 - htb;

wire signed [DATABIT-1:0] hhx;
wire rv_h;

mult16x16_2int M1(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(xt),
	.multiplicator2(ht1subhtb),
	.result_valid(rv_h),
	.result(hhx)
);


mult16x16_2int M2(
	.clk(clk),
	.rst_n(rst_n),
	.en(rv_h&rv_z),
	.multiplicator1(zt1zt),
	.multiplicator2(hhx),
	.result_valid(result_valid),
	.result(result)
);



endmodule