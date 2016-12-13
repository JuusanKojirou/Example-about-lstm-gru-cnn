module cal_r_z(
	input clk,
	input rst_n,
	input en,
	input [63:0] xt,
	input [63:0] ht1,
	input [63:0] wxr_in,
	input [63:0] whr_in,
	input signed [15:0] br,
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

// take 6 cycles


// generate
// 	genvar iwx;
// 	for (iwx = 0; iwx < INPUTDIMEN; iwx=iwx+1) begin: wx
// 		always @(posedge clk or negedge rst_n) begin
// 			if(~rst_n) begin
// 				wxr[iwx] <= 16'd0;
// 			end else begin
// 				wxr[iwx] <= wxr_in[(DATABIT*iwx+(DATABIT-1)):(DATABIT*iwx)];
// 			end
// 		end
// 	end
// endgenerate

// generate
// 	genvar iwh;
// 	for (iwh = 0; iwh < CELLNUM; iwh=iwh+1) begin: wh
// 		always @(posedge clk or negedge rst_n) begin
// 			if(~rst_n) begin
// 				whr[iwh] <= 16'd0;
// 			end else begin
// 				whr[iwh] <= whr_in[(DATABIT*iwh+(DATABIT-1)):(DATABIT*iwh)];
// 			end
// 		end
// 	end
// endgenerate

wire signed [DATABIT-1:0] wxrxt[0:INPUTFOURNUM];
wire rv_wxrxt[0:INPUTFOURNUM-1];
generate
	genvar i;
	for (i = 0; i < INPUTFOURNUM; i=i+1) begin: wxrxt1
		vector_mult_4_parallel wxrxt_m(
			.clk(clk),
			.rst_n(rst_n),
			.en(en),
			.a_0(wxr_in[(DATABIT*(0+i*4)+(DATABIT-1)):(DATABIT*(0+i*4))]),
			.a_1(wxr_in[(DATABIT*(1+i*4)+(DATABIT-1)):(DATABIT*(1+i*4))]),
			.a_2(wxr_in[(DATABIT*(2+i*4)+(DATABIT-1)):(DATABIT*(2+i*4))]),
			.a_3(wxr_in[(DATABIT*(3+i*4)+(DATABIT-1)):(DATABIT*(3+i*4))]),
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

wire signed [DATABIT-1:0] whrht1[0:CELLFOURNUM];
wire rv_whrht1[0:INPUTFOURNUM-1];
generate
	genvar i1;
	for (i1 = 0; i1 < CELLFOURNUM ; i1=i1+1)
	begin: whrht1_m
		vector_mult_4_parallel whrht1_m(
			.clk(clk),
			.rst_n(rst_n),
			.en(en),
			.a_0(whr_in[(DATABIT*(0+i1*4)+(DATABIT-1)):(DATABIT*(0+i1*4))]),
			.a_1(whr_in[(DATABIT*(1+i1*4)+(DATABIT-1)):(DATABIT*(1+i1*4))]),
			.a_2(whr_in[(DATABIT*(2+i1*4)+(DATABIT-1)):(DATABIT*(2+i1*4))]),
			.a_3(whr_in[(DATABIT*(3+i1*4)+(DATABIT-1)):(DATABIT*(3+i1*4))]),
			.b_0(ht1[(DATABIT*(0+i1*4)+15):(DATABIT*(0+i1*4))]),
			.b_1(ht1[(DATABIT*(1+i1*4)+15):(DATABIT*(1+i1*4))]),
			.b_2(ht1[(DATABIT*(2+i1*4)+15):(DATABIT*(2+i1*4))]),
			.b_3(ht1[(DATABIT*(3+i1*4)+15):(DATABIT*(3+i1*4))]),
			.result_valid(rv_whrht1[i1]),
			.ab_out(whrht1[i1])
		);
	end
endgenerate
assign whrht1[CELLFOURNUM] = whrht1[0];//!!!!


reg signed [DATABIT-1:0] rtp;

wire signed [DATABIT-1:0] htb;
always@(*)begin
	rtp = wxrxt[INPUTFOURNUM] + whrht1[CELLFOURNUM] + br;
end

sigmoid s1(
	.clk(clk),
	.rst_n(rst_n),
	.en(rv_wxrxt[0]&rv_whrht1[0]),
	.sigmoid_input(rtp),
	.result_valid(result_valid),
	.sigmoid_output(result)
);



endmodule