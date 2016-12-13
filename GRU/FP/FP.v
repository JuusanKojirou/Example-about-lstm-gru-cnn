`timescale 1ns / 1ps

module FP(
	input clk,
	input rst_n,
	input load_weight,
	input start,
	input [7:0] step,
	input [63:0] xt,
	input [63:0] w1,
	input [63:0] w2,
	output result_valid,
	output result
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

reg clkl;
reg [5:0] count_clkl;

always @(posedge clk or negedge rst_n) begin
	if(~rst_n) begin
		clkl <= 0;
		count_clkl <= 0;
	end else begin
		if(count_clkl==6'd11) begin
			clkl <= ~clkl;
			count_clkl <= 0;
		end else begin
			clkl <= clkl;
			count_clkl <= count_clkl+6'd1;
		end
	end
end

//reg finish;
// reg start_in;
// reg startd1;

// always @(posedge clkl or negedge rst_n) begin
// 	if(~rst_n) begin
// 		startd1 <= 0;
// 		start_in <= 0;
// 	end else begin
// 		startd1 <= start;
// 		start_in <= startd1;
// 	end
// end

reg [3:0] add;
wire [95:0] weight;

reg [HWXNUM-1:0] wxr;
reg [HWXNUM-1:0] wxz;
reg [HWXNUM-1:0] wxh;
reg [HWHNUM-1:0] whr;
reg [HWHNUM-1:0] whz;
reg [HWHNUM-1:0] whh;


always @(posedge clkl or negedge rst_n) begin
	if(~rst_n) begin
		add <= 0;
		wxr <= 0;
		wxz <= 0;
		wxh <= 0;
		whr <= 0;
		whz <= 0;
		whh <= 0;
	end else begin
		if(load_weight) begin
			if(add==INPUTDIMEN*CELLNUM)begin
				add <= 0;
				wxr <= wxr;
				wxz <= wxz;
				wxh <= wxh;
				whr <= whr;
				whz <= whz;
				whh <= whh;
			end
			else begin
				add <= add+4'd1;
				whh <= (whh<<DATABIT) + weight[DATABIT-1:0];
				whz <= (whz<<DATABIT) + weight[DATABIT*2-1:DATABIT];
				whr <= (whr<<DATABIT) + weight[DATABIT*3-1:DATABIT*2];
				wxh <= (wxh<<DATABIT) + weight[DATABIT*4-1:DATABIT*3];
				wxz <= (wxz<<DATABIT) + weight[DATABIT*5-1:DATABIT*4];
				wxr <= (wxr<<DATABIT) + weight[DATABIT*6-1:DATABIT*5];
			end
		end
		else begin
			add <= 0;
			wxr <= wxr;
			wxz <= wxz;
			wxh <= wxh;
			whr <= whr;
			whz <= whz;
			whh <= whh;
		end
	end
end


mem96x16 mem(
	.address(add),
	.clock(clk),
	.data(),
	.wren(0),
	.q(weight)
	);

wire [HTNUM-1:0] h;
wire rv_hl;
hidden_layer hl(
	.clk(clk),
	.clk_18(clkl),
	.rst_n(rst_n),
	.en(start),
	.xt(xt),
	.wxr(wxr),
	.wxz(wxz),
	.wxh(wxh),
	.whr(whr),
	.whz(whz),
	.whh(whh),
	.br(16'd0),
	.bz(16'd0),
	.bh(16'd0),
	.result_valid(rv_hl),
	.h(h)
	);

wire [HTNUM-1:0] ave_out;
wire rv_av[0:CELLNUM-1];
generate
	genvar i;
	for (i = 0; i < CELLNUM; i=i+1) begin: h_ave
		average hav(
			.clk(clk),
			.clkl(clkl),
			.rst_n(rst_n),
			.ht(h[(DATABIT*(i+1)-1):(DATABIT*i)]),
			.start(rv_hl),
			.t(step),
			.done_sig(rv_av[i]),
			.ave_out(ave_out[(DATABIT*(i+1)-1):(DATABIT*i)])
		);
	end
endgenerate

main m(
	.clk(clk),
	.rst_n(rst_n),
	.en(rv_av[0]),
	.w11(w1[15:0]),
	.w12(w1[31:16]),
	.w13(w1[47:32]),
	.w14(w1[63:48]),
	.w21(w2[15:0]),
	.w22(w2[31:16]),
	.w23(w2[47:32]),
	.w24(w2[63:48]),
	.h1(ave_out[15:0]),
	.h2(ave_out[31:16]),
	.h3(ave_out[47:32]),
	.h4(ave_out[63:48]),
	.result_valid(result_valid),
	.pred(result)
	);

endmodule
