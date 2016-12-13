`include "parameter"
module formula4(
	input clk,
	input rst_n,
	input en,
	input signed [DATABIT-1:0] htb,
	input [HTNUM-1:0] rt,
	input [HTNUM-1:0] ht1,
	input [HTNUM-1:0] whh,
	input signed [DATABIT-1:0] rt0_w,
	input signed [DATABIT-1:0] rt1_w,
	input signed [DATABIT-1:0] rt2_w,
	input signed [DATABIT-1:0] rt3_w,
	input signed [DATABIT-1:0] dh0_dw,
	input signed [DATABIT-1:0] dh1_dw,
	input signed [DATABIT-1:0] dh2_dw,
	input signed [DATABIT-1:0] dh3_dw,	
	output wire result_valid,
	output wire signed [DATABIT-1:0] htb_w
);

//take 9 clk cycles

wire signed [DATABIT-1:0] htb2;
wire rv_htb2;
mult16x16_2int M_htb2(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(htb),
	.multiplicator2(htb),
	.result_valid(rv_htb2),
	.result(htb2)
	);

wire signed [DATABIT-1:0] onesubhtb2;
assign onesubhtb2 = ~htb2 + 16'b0100000000000001;

reg [HTNUM-1:0] whh1,whh2,whh3;
always @(posedge clk or negedge rst_n) begin
	if(~rst_n) begin
		whh1 <= 0;
		whh2 <= 0;
		whh3 <= 0;
	end else begin
		whh1 <= whh;
		whh2 <= whh1;
		whh3 <= whh2;
	end
end

wire signed [DATABIT-1:0] whhrt[CELLNUM-1:0];
wire rv_whhrt_0;
wire rv_whhrt_1;
wire rv_whhrt_2;
wire rv_whhrt_3;

mult16x16_2int whhrt_0(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(dh0_dw),
	.multiplicator2(rt[15:0]),
	.result_valid(rv_whhrt_0),
	.result(whhrt[0])
	);
mult16x16_2int whhrt_1(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(dh1_dw),
	.multiplicator2(rt[31:16]),
	.result_valid(rv_whhrt_1),
	.result(whhrt[1])
	);
mult16x16_2int whhrt_2(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(dh2_dw),
	.multiplicator2(rt[47:32]),
	.result_valid(rv_whhrt_2),
	.result(whhrt[2])
	);
mult16x16_2int whhrt_3(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(dh3_dw),
	.multiplicator2(rt[63:48]),
	.result_valid(rv_whhrt_3),
	.result(whhrt[3])
	);

wire signed [DATABIT-1:0] temp1;
wire rv_vm1;
vector_mult_4_parallel VM1(
	.clk(clk),
	.rst_n(rst_n),
	.en(rv_whhrt_0),
	.a_0(whhrt[0]),
	.a_1(whhrt[1]),
	.a_2(whhrt[2]),
	.a_3(whhrt[3]),
	.b_0(whh3[15:0]),
	.b_1(whh3[31:16]),
	.b_2(whh3[47:32]),
	.b_3(whh3[63:48]),
	.result_valid(rv_vm1),
	.ab_out(temp1)
);

wire signed [DATABIT-1:0] whhht1[CELLNUM-1:0];
wire rv_whhht_0;
wire rv_whhht_1;
wire rv_whhht_2;
wire rv_whhht_3;

mult16x16_2int whhht1_0(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(rt0_w),
	.multiplicator2(ht1[15:0]),
	.result_valid(rv_whhht_0),
	.result(whhht1[0])
	);
mult16x16_2int whhht1_1(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(rt1_w),
	.multiplicator2(ht1[31:16]),
	.result_valid(rv_whhht_1),
	.result(whhht1[1])
	);
mult16x16_2int whhht1_2(
	.clk(clk),
	.rst_n(rst_n),
	.multiplicator1(rt2_w),
	.multiplicator2(ht1[47:32]),
	.result_valid(rv_whhht_2),
	.result(whhht1[2])
	);
mult16x16_2int whhht1_3(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(rt3_w),
	.multiplicator2(ht1[63:48]),
	.result_valid(rv_whhht_3),
	.result(whhht1[3])
	);

wire signed [DATABIT-1:0] temp2;
wire rv_vm2;
vector_mult_4_parallel VM2(
	.clk(clk),
	.rst_n(rst_n),
	.en(rv_whhht_0),
	.a_0(whhht1[0]),//<<<2
	.a_1(whhht1[1]),//<<<2
	.a_2(whhht1[2]),//<<<2
	.a_3(whhht1[3]),//<<<2
	.b_0(whh3[15:0]),//<<<5
	.b_1(whh3[31:16]),//<<<5
	.b_2(whh3[47:32]),//<<<5
	.b_3(whh3[63:48]),//<<<5
	.result_valid(rv_vm2),
	.ab_out(temp2)
);

wire signed [DATABIT-1:0] temp3;
assign temp3 = temp1 + temp2;

// reg [DATABIT-1:0] osh2d1,osh2d2,osh2d3;
// always @(posedge clk or negedge rst_n) begin
// 	if(~rst_n) begin
// 		osh2d1 <= 0;
// 		osh2d2 <= 0;
// 		osh2d3 <= 0;
// 	end else begin
// 		osh2d1 <= onesubhtb2;
// 		osh2d2 <= osh2d1;
// 		osh2d3 <= osh2d2;
// 	end
// end

mult16x16_2int FF(
	.clk(clk),
	.rst_n(rst_n),
	.en(rv_vm1),
	.multiplicator1(onesubhtb2),
	.multiplicator2(temp3),
	.result_valid(result_valid),
	.result(htb_w)
	);


endmodule