//`include "parameter"
module formula1(
	input clk,
	input rst_n,
	input en,
	input signed [DATABIT-1:0] zt_w,
	input signed [DATABIT-1:0] ht1,
	input signed [DATABIT-1:0] htb,
	input signed [DATABIT-1:0] zt,
	input signed [DATABIT-1:0] ht1_w,
	input signed [DATABIT-1:0] htb_w,
	output wire result_valid,
	output wire signed [DATABIT-1:0] ht_w
);

// take 3 clk cycles

wire signed [DATABIT-1:0] ht1subhtb;
assign ht1subhtb = ht1 - htb;

wire signed [DATABIT-1:0] onesubzt;
assign onesubzt = ~zt + 16'b0100000000000001;

wire signed [DATABIT-1:0] temp1;
wire signed [DATABIT-1:0] temp2;
wire signed [DATABIT-1:0] temp3;

wire rv1;
wire rv2;
wire rv3;

mult16x16_2int M1(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(zt_w),
	.multiplicator2(ht1subhtb),
	.result_valid(rv1),
	.result(temp1)
	);

mult16x16_2int M2(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(zt),
	.multiplicator2(ht1_w),
	.result_valid(rv2),
	.result(temp2)
	);

mult16x16_2int M3(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(htb_w),
	.multiplicator2(onesubzt),
	.result_valid(rv3),
	.result(temp3)
	);


	assign result_valid = rv1&rv2&rv3;
	assign ht_w = temp1 + temp2 +temp3;


endmodule