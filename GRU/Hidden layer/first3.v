//`include "parameter.v"
module first3(
	input clk,
	//input clk_18,
	input rst_n,
	input en,
	input signed [WNUM-1:0] whr_in,
	input signed [WNUM-1:0] whz_in,
	input signed [WNUM-1:0] whh_in,
	input signed [HTNUM-1:0] htb,
	input signed [HTNUM-1:0] ht1,
	input signed [HTNUM-1:0] zt,
	input signed [HTNUM-1:0] rt,
	input signed [DATABIT-1:0] dh0_dw,
	input signed [DATABIT-1:0] dh1_dw,
	input signed [DATABIT-1:0] dh2_dw,
	input signed [DATABIT-1:0] dh3_dw,
	output wire result_valid,
	output wire signed [DATABIT-1:0] result_0,
	output wire signed [DATABIT-1:0] result_1,
	output wire signed [DATABIT-1:0] result_2,
	output wire signed [DATABIT-1:0] result_3
);

wire [HTNUM-1:0] dh_dw;
assign dh_dw = {dh3_dw,dh2_dw,dh1_dw,dh0_dw};

wire signed [DATABIT-1:0] zt_wxr[CELLNUM-1:0];
wire signed [DATABIT-1:0] ht_wxr[CELLNUM-1:0];
wire signed [DATABIT-1:0] htb_wxr[CELLNUM-1:0];
wire signed [DATABIT-1:0] rt_wxr[CELLNUM-1:0];

wire rv_zt_wxr[CELLNUM-1:0];
wire rv_ht_wxr[CELLNUM-1:0];
wire rv_htb_wxr[CELLNUM-1:0];
wire rv_rt_wxr[CELLNUM-1:0];


generate
	genvar idd;
	for (idd = 0; idd < CELLNUM; idd=idd+1) begin:calculate_dz_dw
		d_d dz_dw(
			.clk(clk), 
			.rst_n(rst_n),
			.en(en),
			.rorz(zt[((idd+1)*DATABIT-1):(idd*DATABIT)]), 
			.w(whz_in[((idd+1)*HTNUM-1):(idd*HTNUM)]), 
			.dh0_dw(dh0_dw),
			.dh1_dw(dh1_dw),
			.dh2_dw(dh2_dw),
			.dh3_dw(dh3_dw),
			.result_valid(rv_zt_wxr[idd]),
			.out(zt_wxr[idd])
		);
	end
endgenerate

generate
	genvar iddx;
	for (iddx = 0; iddx < CELLNUM; iddx=iddx+1) begin:calculate_dr_dw
		d_d dr_dw(
			.clk(clk), 
			.rst_n(rst_n),
			.en(en),
			.rorz(rt[((iddx+1)*DATABIT-1):(iddx*DATABIT)]),
			.w(whr_in[((iddx+1)*HTNUM-1):(iddx*HTNUM)]),
			.dh0_dw(dh0_dw),
			.dh1_dw(dh1_dw),
			.dh2_dw(dh2_dw),
			.dh3_dw(dh3_dw),
			.result_valid(rv_rt_wxr[iddx]),
			.out(rt_wxr[iddx])
		);
	end
endgenerate
//6 clk

wire en_f4;
assign en_f4 = rv_rt_wxr[0]&rv_rt_wxr[1]&rv_rt_wxr[2]&rv_rt_wxr[3];

generate
	genvar if4;
	for (if4 = 0; if4 < CELLNUM; if4=if4+1) begin:calculate_dhtb_dw
		formula4 dhtb_dw(
			.clk(clk),
			.rst_n(rst_n),
			.en(en_f4),
			.htb(htb[((if4+1)*DATABIT-1):(if4*DATABIT)]),
			.rt(rt),
			.ht1(ht1),
			.whh(whh_in[((if4+1)*HTNUM-1):(if4*HTNUM)]),
			.rt0_w(rt_wxr[0]),
			.rt1_w(rt_wxr[1]),
			.rt2_w(rt_wxr[2]),
			.rt3_w(rt_wxr[3]),
			.dh0_dw(dh0_dw),
			.dh1_dw(dh1_dw),
			.dh2_dw(dh2_dw),
			.dh3_dw(dh3_dw),
			.result_valid(rv_htb_wxr[if4]),
			.htb_w(htb_wxr[if4])
		);
	end
endgenerate
// 6+9

generate
	genvar if1;
	for (if1 = 0; if1 < CELLNUM; if1=if1+1) begin:calculate_dht_dw
		formula1 dht_dw(
			.clk(clk), 
			.rst_n(rst_n), 
			.en(rv_htb_wxr[if1]),
			.zt_w(zt_wxr[if1]),
			.ht1(ht1[((if1+1)*DATABIT-1):(if1*DATABIT)]),
			.htb(htb[((if1+1)*DATABIT-1):(if1*DATABIT)]),
			.zt(zt[((if1+1)*DATABIT-1):(if1*DATABIT)]),
			.ht1_w(dh_dw[((if1+1)*DATABIT-1):(if1*DATABIT)]),
			.htb_w(htb_wxr[if1]),
			.ht_w(ht_wxr[if1])
		);
	end
endgenerate
//6+9+3


// always @(posedge clk_18 or negedge rst_n) begin
// 	if(~rst_n) begin
// 		result_0 <= 0;
// 		result_1 <= 0;
// 		result_2 <= 0;
// 		result_3 <= 0;
// 	end else begin
// 		result_0 <= ht_wxr[0];
// 		result_1 <= ht_wxr[1];
// 		result_2 <= ht_wxr[2];
// 		result_3 <= ht_wxr[3];
// 	end
// end

assign result_0 = ht_wxr[0];
assign result_1 = ht_wxr[1];
assign result_2 = ht_wxr[2];
assign result_3 = ht_wxr[3];

//output

endmodule