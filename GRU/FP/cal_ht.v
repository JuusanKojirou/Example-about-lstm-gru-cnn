module cal_ht(
	input clk,
	input rst_n,
	input en,
	input signed [15:0] zt,
	input signed [15:0] ht1,
	input signed [15:0] htb,
	output result_valid,
	output signed [15:0] ht
);
parameter DATABIT = 16;
parameter INPUTDIMEN = 4;
parameter CELLNUM = 4;
parameter STEP = 10;

parameter XTNUM = INPUTDIMEN*DATABIT;
parameter HTNUM = CELLNUM*DATABIT;
parameter HWXNUM = INPUTDIMEN*CELLNUM*DATABIT;
parameter HWHNUM = CELLNUM*CELLNUM*DATABIT;

parameter INPUTFOURNUM = 1;
parameter CELLFOURNUM = 1;

//take 3 clk cycles

wire signed [DATABIT-1:0] ztht1, oneszt, zthtb;

assign oneszt = ~zt + 16'b0100000000000001;

wire rv_m1, rv_m2;
mult16x16_2int ztht1_m(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(zt),
	.multiplicator2(ht1),
	.result_valid(rv_m1),
	.result(ztht1)
	);
mult16x16_2int zthtb_1(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(oneszt),
	.multiplicator2(htb),
	.result_valid(rv_m2),
	.result(zthtb)
	);

assign ht = ztht1 + zthtb;
assign result_valid = rv_m1&rv_m2;

endmodule