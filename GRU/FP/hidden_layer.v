//`include "parameter.v"
module hidden_layer (
	input clk,    
	input clk_18,
	input rst_n,
	input en,
	input [63:0] xt,
	input [255:0] wxr,
	input [255:0] wxz,
	input [255:0] wxh,
	input [255:0] whr,
	input [255:0] whz,
	input [255:0] whh,
	input [15:0] br,
	input [15:0] bz,
	input [15:0] bh,
	output reg result_valid,
	output reg [63:0] h
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

wire signed [DATABIT-1:0] rt[0:CELLNUM-1];
wire rv_rt[0:CELLNUM-1];
generate 
	genvar irt;
	for (irt = 0; irt < CELLNUM; irt=irt+1) begin:calculate_rt
		cal_r_z r_u(
			.clk(clk),
			.rst_n(rst_n),
			.en(en),
			.xt(xt),
			.ht1(h),
			.wxr_in(wxr[(XTNUM*irt+(XTNUM-1)):(XTNUM*irt)]),
			.whr_in(whr[(HTNUM*irt+(HTNUM-1)):(HTNUM*irt)]),
			.br(br),
			.result_valid(rv_rt[irt]),
			.result(rt[irt])
		);
	end
endgenerate
//rt, 8

wire signed [DATABIT-1:0] zt[0:CELLNUM-1];
wire rv_zt[0:CELLNUM-1];
generate 
	genvar izt;
	for (izt = 0; izt < CELLNUM; izt=izt+1) begin:calculate_zt
		cal_r_z z_u(
			.clk(clk),
			.rst_n(rst_n),
			.en(en),
			.xt(xt),
			.ht1(h),
			.wxr_in(wxz[(XTNUM*izt+(XTNUM-1)):(XTNUM*izt)]),
			.whr_in(whz[(HTNUM*izt+(HTNUM-1)):(HTNUM*izt)]),
			.br(bz),
			.result_valid(rv_zt[izt]),
			.result(zt[izt])
		);
	end
endgenerate
//zt, 8

wire [HTNUM-1:0] rt_long;
assign rt_long = {rt[3],rt[2],rt[1],rt[0]};

wire signed [DATABIT-1:0] htb[0:CELLNUM-1];
wire rv_htb[0:CELLNUM-1];
generate 
	genvar ihtb;
	for (ihtb = 0; ihtb < CELLNUM; ihtb=ihtb+1) begin:calculate_htb
		cal_htb htb_u(
			.clk(clk),
			.rst_n(rst_n),
			.en(rv_rt[ihtb]),
			.xt(xt),
			.ht1(h),
			.rt(rt_long),
			.wxh_in(wxh[(XTNUM*ihtb+(XTNUM-1)):(XTNUM*ihtb)]),
			.whh_in(whh[(HTNUM*ihtb+(HTNUM-1)):(HTNUM*ihtb)]),
			.bh(bh),
			.result_valid(rv_htb[ihtb]),
			.result(htb[ihtb])
		);
	end
endgenerate
//htb, 19

wire signed [DATABIT-1:0] ht[0:CELLNUM-1];
wire rv_ht[0:CELLNUM-1];
generate 
	genvar iht;
	for (iht = 0; iht < CELLNUM; iht=iht+1) begin:calculate_ht
		cal_ht cht(
			.clk(clk),
			.rst_n(rst_n),
			.en(rv_htb[iht]),
			.zt(zt[iht]),
			.ht1(h[(DATABIT*(iht+1)-1):(DATABIT*iht)]),
			.htb(htb[iht]),
			.result_valid(rv_ht[iht]),
			.ht(ht[iht])
		);
	end
endgenerate
//ht, 22

generate
	genvar ih;
	for (ih = 0; ih < CELLNUM; ih=ih+1) begin: finall
		always @(posedge clk_18 or negedge rst_n) begin
			if(~rst_n) begin
				h[(DATABIT*(ih+1)-1):(DATABIT*ih)] <= 16'd0;
			end else begin
				h[(DATABIT*(ih+1)-1):(DATABIT*ih)] <= ht[ih];
			end
		end
	end
endgenerate

always @(posedge clk_18 or negedge rst_n) begin
	if(~rst_n) begin
		result_valid <= 0;
	end else begin
		result_valid <= rv_ht[0]&rv_ht[1]&rv_ht[2]&rv_ht[3];
	end
end

endmodule