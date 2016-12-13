//`include "parameter.v"
module dh_dwxh(
	input clk,
	input clk_18,
	input rst_n,
	input en,
	input [1:0] n,
	input signed [DATABIT-1:0] xt_in,//m=n; m!=n,xt=0
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
	output reg result_valid,
	output reg signed [DATABIT-1:0] result_0,
	output reg signed [DATABIT-1:0] result_1,
	output reg signed [DATABIT-1:0] result_2,
	output reg signed [DATABIT-1:0] result_3
);

wire signed [DATABIT-1:0] f3_0;
wire signed [DATABIT-1:0] f3_1;
wire signed [DATABIT-1:0] f3_2;
wire signed [DATABIT-1:0] f3_3;

wire rv_f3;

first3 F3(
	.clk(clk),
	//.clk_18(clk_18),
	.rst_n(rst_n),
	.en(en),
	.whr_in(whr_in),
	.whz_in(whz_in),
	.whh_in(whh_in),
	.htb(htb),
	.ht1(ht1),
	.zt(zt),
	.rt(rt),
	.dh0_dw(dh0_dw),
	.dh1_dw(dh1_dw),
	.dh2_dw(dh2_dw),
	.dh3_dw(dh3_dw),
	.result_valid(rv_f3),
	.result_0(f3_0),
	.result_1(f3_1),
	.result_2(f3_2),
	.result_3(f3_3)
);

reg signed [DATABIT-1:0] htbm,ztm;
always@(*) begin
	case (n)
		2'b00: htbm = htb[15:0];
		2'b01: htbm = htb[31:16];
		2'b10: htbm = htb[47:32];
		2'b11: htbm = htb[63:48];
		default: htbm = 16'b0;
	endcase
end

always@(*) begin
	case (n)
		2'b00: ztm = zt[15:0];
		2'b01: ztm = zt[31:16];
		2'b10: ztm = zt[47:32];
		2'b11: ztm = zt[63:48];
		default: ztm = 16'b0;
	endcase
end

wire rv_a1;
wire signed [DATABIT-1:0] a1_result;

A_2 A2(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.xt(xt_in),
	.htb(htbm),
	.zt(ztm),
	.result_valid(rv_a1),
	.result(a1_result)
);

always @(posedge clk_18 or negedge rst_n) begin
	if(~rst_n) begin
		result_0 <= 0;
		result_1 <= 0;
		result_2 <= 0;
		result_3 <= 0;
		result_valid <= 0;
	end 
	else begin
		result_valid <= rv_f3 & rv_a1;
		if(n==2'b00) begin
			result_0 <= f3_0 + a1_result;
			result_1 <= f3_1;
			result_2 <= f3_2;
			result_3 <= f3_3;
		end
		else if(n==2'b01) begin
			result_0 <= f3_0;
			result_1 <= f3_1 + a1_result;
			result_2 <= f3_2;
			result_3 <= f3_3;
		end
		else if(n==2'b10) begin
			result_0 <= f3_0;
			result_1 <= f3_1;
			result_2 <= f3_2 + a1_result;
			result_3 <= f3_3;
		end
		else if(n==2'b11) begin
			result_0 <= f3_0;
			result_1 <= f3_1;
			result_2 <= f3_2;
			result_3 <= f3_3 + a1_result;
		end
		else begin
			result_0 <= 0;
			result_1 <= 0;
			result_2 <= 0;
			result_3 <= 0;
		end
	end
end

endmodule // dh_dwxh