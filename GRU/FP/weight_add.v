`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    19:37:26 12/18/2015 
// Design Name: 
// Module Name:    weight_add 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module weight_add(
	input clk,
    input rst_n,
    input en,
    input [15:0] w11,
    input [15:0] w12,
    input [15:0] w13,
	input [15:0] w14,
	input [15:0] w21,
    input [15:0] w22,
    input [15:0] w23,
	input [15:0] w24,
    input [15:0] h1,
    input [15:0] h2,
    input [15:0] h3,
	input [15:0] h4,
	output result_valid,
    output [15:0] out1,
    output [15:0] out2
    );
	 
wire [15:0]	temp11;
wire [15:0] temp12;
wire [15:0] temp13;
wire [15:0] temp14;
wire [15:0]	temp21;
wire [15:0] temp22;
wire [15:0] temp23;
wire [15:0] temp24;
wire [15:0]	temp1;
wire [15:0]	temp2;


wire rv_11;
wire rv_12;
wire rv_13;
wire rv_14;
wire rv_21;
wire rv_22;
wire rv_23;
wire rv_24;

	 
mult16x16_2int s11(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(w11),
	.multiplicator2(h1),
	.result_valid(rv_11),
	.result(temp11)
);
							 
mult16x16_2int s12(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(w12),
	.multiplicator2(h2),
	.result_valid(rv_12),
	.result(temp12)
);

mult16x16_2int s13(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(w13),
	.multiplicator2(h3),
	.result_valid(rv_13),
	.result(temp13)
);		
							
mult16x16_2int s14(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(w14),
	.multiplicator2(h4),
	.result_valid(rv_14),
	.result(temp14)
);

mult16x16_2int s21(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(w21),
	.multiplicator2(h1),
	.result_valid(rv_21),
	.result(temp21)
);
							 
mult16x16_2int s22(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(w22),
	.multiplicator2(h2),
	.result_valid(rv_22),
	.result(temp22)
);

mult16x16_2int s23(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(w23),
	.multiplicator2(h3),
	.result_valid(rv_23),
	.result(temp23)
);	

mult16x16_2int s24(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(w24),
	.multiplicator2(h4),
	.result_valid(rv_24),
	.result(temp24)
);								 

assign out1 = temp11 + temp12 + temp13 + temp14;	
assign out2 = temp21 + temp22 + temp23 + temp24;	
assign result_valid = rv_11;


endmodule
