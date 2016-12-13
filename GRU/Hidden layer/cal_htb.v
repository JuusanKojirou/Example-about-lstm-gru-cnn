module cal_htb(
	input clk,
	input rst_n,
	input en,
	input [63:0] xt,
	input [63:0] ht1,
	input [63:0] rt,
	input [63:0] wxh_in,
	input [63:0] whh_in,
	input signed [15:0] bh,
	output result_valid,
	output signed [15:0] result
);
parameter INPUTDIMEN = 4;
parameter CELLNUM = 4;
parameter DATABIT = 16;
parameter STEP = 10;

parameter XTNUM = INPUTDIMEN*DATABIT;
parameter HTNUM = CELLNUM*DATABIT;
parameter HWXNUM = INPUTDIMEN*CELLNUM*DATABIT;
parameter HWHNUM = CELLNUM*CELLNUM*DATABIT;

parameter INPUTFOURNUM = 1;
parameter CELLFOURNUM = 1;

// take 11 cycles


wire signed [DATABIT-1:0] wxrxt[0:INPUTFOURNUM];
wire rv_wxrxt[0:INPUTFOURNUM-1];
generate
	genvar i;
	for (i = 0; i < INPUTFOURNUM; i=i+1) begin: wxrxt1
		vector_mult_4_parallel wxrxt_m(
			.clk(clk),
			.rst_n(rst_n),
			.en(en),
			.a_0(wxh_in[(DATABIT*(0+i*4)+(DATABIT-1)):(DATABIT*(0+i*4))]),
			.a_1(wxh_in[(DATABIT*(1+i*4)+(DATABIT-1)):(DATABIT*(1+i*4))]),
			.a_2(wxh_in[(DATABIT*(2+i*4)+(DATABIT-1)):(DATABIT*(2+i*4))]),
			.a_3(wxh_in[(DATABIT*(3+i*4)+(DATABIT-1)):(DATABIT*(3+i*4))]),
			.b_0(xt[(DATABIT*(0+i*4)+(DATABIT-1)):(DATABIT*(0+i*4))]),
			.b_1(xt[(DATABIT*(1+i*4)+(DATABIT-1)):(DATABIT*(1+i*4))]),
			.b_2(xt[(DATABIT*(2+i*4)+(DATABIT-1)):(DATABIT*(2+i*4))]),
			.b_3(xt[(DATABIT*(3+i*4)+(DATABIT-1)):(DATABIT*(3+i*4))]),
			.result_valid(rv_wxrxt[i]),
			.ab_out(wxrxt[i])
		);
	end
endgenerate
assign wxrxt[INPUTFOURNUM] = wxrxt[0]; //!!!!

wire signed [DATABIT-1:0] rtht1[0:CELLNUM-1];
wire rv_rtht1[0:CELLNUM-1];
generate
	genvar ii;
	for (ii = 0; ii < CELLNUM; ii=ii+1) begin:rt_ht1
		mult16x16_2int rtht1_m(
			.clk(clk),
			.rst_n(rst_n),
			.en(en),
			.multiplicator1(rt[(DATABIT*(ii+1)-1):(DATABIT*ii)]),
			.multiplicator2(ht1[(DATABIT*(ii+1)-1):(DATABIT*ii)]),
			.result_valid(rv_rtht1[ii]),
			.result(rtht1[ii])
		);
	end
endgenerate


wire signed [DATABIT-1:0] whrht1[0:CELLFOURNUM];
wire rv_whrht1[0:INPUTFOURNUM-1];
generate
	genvar i1;
	for (i1 = 0; i1 < CELLFOURNUM ; i1=i1+1)
	begin: whrht1_m
		vector_mult_4_parallel whrht1_m(
			.clk(clk),
			.rst_n(rst_n),
			.en(rv_rtht1[0]),
			.a_0(whh_in[(DATABIT*(0+i1*4)+(DATABIT-1)):(DATABIT*(0+i1*4))]),
			.a_1(whh_in[(DATABIT*(1+i1*4)+(DATABIT-1)):(DATABIT*(1+i1*4))]),
			.a_2(whh_in[(DATABIT*(2+i1*4)+(DATABIT-1)):(DATABIT*(2+i1*4))]),
			.a_3(whh_in[(DATABIT*(3+i1*4)+(DATABIT-1)):(DATABIT*(3+i1*4))]),
			.b_0(rtht1[i1*4+0]),
			.b_1(rtht1[i1*4+1]),
			.b_2(rtht1[i1*4+2]),
			.b_3(rtht1[i1*4+3]),
			.result_valid(rv_whrht1[i1]),
			.ab_out(whrht1[i1])
		);
	end
endgenerate
assign whrht1[CELLFOURNUM] = whrht1[0];//!!!!


reg signed [DATABIT-1:0] rtp;

always@(*)begin
	rtp = wxrxt[INPUTFOURNUM] + whrht1[CELLFOURNUM] + bh;
end


tanh t1(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.tanh_input(rtp),
	.result_valid(result_valid),
	.tanh_output(result)
);


endmodule