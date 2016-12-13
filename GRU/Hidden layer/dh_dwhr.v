//`include "parameter"
module dh_dwhr(
	input clk,
	input clk_18,
	input rst_n,
	input en,
	input [1:0] n,
	input signed [DATABIT-1:0] ht1i,//m=n; m!=n,xt=0
	input [WNUM-1:0] whr_in,
	input [WNUM-1:0] whz_in,
	input [WNUM-1:0] whh_in,
	input [HTNUM-1:0] htb,
	input [HTNUM-1:0] ht1,
	input [HTNUM-1:0] zt,
	input [HTNUM-1:0] rt,
	input signed [DATABIT-1:0] dh0_dw,
	input signed [DATABIT-1:0] dh1_dw,
	input signed [DATABIT-1:0] dh2_dw,
	input signed [DATABIT-1:0] dh3_dw,
	output wire result_valid,
	output reg signed [DATABIT-1:0] result_0,
	output reg signed [DATABIT-1:0] result_1,
	output reg signed [DATABIT-1:0] result_2,
	output reg signed [DATABIT-1:0] result_3
);

// reg [WNUM-1:0]  whhd1, whhd2, whhd3, whhd4, whhd5, whhd6;
// reg [HTNUM-1:0] htbd1, htbd2, htbd3, htbd4, htbd5, htbd6;
// reg [HTNUM-1:0] ht1d1, ht1d2, ht1d3, ht1d4, ht1d5, ht1d6;
// reg [HTNUM-1:0] ztd1, ztd2, ztd3, ztd4, ztd5, ztd6, ztd7, ztd8, ztd9;
// reg [DATABIT-1:0] dh0_dwd1, dh0_dwd2, dh0_dwd3, dh0_dwd4, dh0_dwd5, dh0_dwd9, dh0_dwd6, dh0_dwd7, dh0_dwd8;
// reg [DATABIT-1:0] dh1_dwd1, dh1_dwd2, dh1_dwd3, dh1_dwd4, dh1_dwd5, dh1_dwd9, dh1_dwd6, dh1_dwd7, dh1_dwd8;
// reg [DATABIT-1:0] dh2_dwd1, dh2_dwd2, dh2_dwd3, dh2_dwd4, dh2_dwd5, dh2_dwd9, dh2_dwd6, dh2_dwd7, dh2_dwd8;
// reg [DATABIT-1:0] dh3_dwd1, dh3_dwd2, dh3_dwd3, dh3_dwd4, dh3_dwd5, dh3_dwd9, dh3_dwd6, dh3_dwd7, dh3_dwd8;
// reg [HTNUM-1:0] rtd1, rtd2, rtd3, rtd4, rtd5, rtd6;

// always @(posedge clk or negedge rst_n) begin
// 	if(~rst_n) begin
// 		whhd
// 	end else begin
// 		 <= ;
// 	end
// end
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

// reg signed [HTNUM-1:0] whr[CELLNUM-1:0];
// reg signed [HTNUM-1:0] whz[CELLNUM-1:0];
// reg signed [HTNUM-1:0] whh[CELLNUM-1:0];

wire signed [DATABIT-1:0] xt[CELLNUM-1:0];

//改数据bit或者CELL输的时候记得改这里！
assign xt[0] = (n==2'b00)? ht1i:16'd0;
assign xt[1] = (n==2'b01)? ht1i:16'd0;
assign xt[2] = (n==2'b10)? ht1i:16'd0;
assign xt[3] = (n==2'b11)? ht1i:16'd0;

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
		d_d_x dr_dw(
			.clk(clk), 
			.rst_n(rst_n),
			.en(en),
			.rorz(rt[((iddx+1)*DATABIT-1):(iddx*DATABIT)]),
			.x_in(xt[iddx]), 
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


always @(posedge clk_18 or negedge rst_n) begin
	if(~rst_n) begin
		result_0 <= 0;
		result_1 <= 0;
		result_2 <= 0;
		result_3 <= 0;
	end else begin
		result_0 <= ht_wxr[0];
		result_1 <= ht_wxr[1];
		result_2 <= ht_wxr[2];
		result_3 <= ht_wxr[3];
	end
end
//output

endmodule