`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    21:11:35 12/18/2015 
// Design Name: 
// Module Name:    main 
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
module main(
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
//	input y,
	output result_valid,
	output pred
	// output reg [15:0] cost_w11,
	// output reg [15:0] cost_w12,
	// output reg [15:0] cost_w13,
	// output reg [15:0] cost_w14,
	// output reg [15:0] cost_w21,
	// output reg [15:0] cost_w22,
	// output reg [15:0] cost_w23,
	// output reg [15:0] cost_w24,
	// output reg [15:0] cost_h1,
	// output reg [15:0] cost_h2,
	// output reg [15:0] cost_h3,
	// output reg [15:0] cost_h4
    );
	 
	 wire [15:0] m1;
	 wire [15:0] m2;
	 // wire [15:0] p1;
	 // wire [15:0] p2;
	 // wire [15:0] cost_x1;
	 // wire [15:0] cost_x2;
	 // wire [15:0] y1;
	 // wire [15:0] y2;
	 // wire [15:0] r11;
	 // wire [15:0] r12;
	 // wire [15:0] r13;
	 // wire [15:0] r14;
	 // wire [15:0] r21;
	 // wire [15:0] r22;
	 // wire [15:0] r23;
	 // wire [15:0] r24;
	 // wire [15:0] z11;
	 // wire [15:0] z21;
	 // wire [15:0] z31;
	 // wire [15:0] z41;
	 // wire [15:0] z12;
	 // wire [15:0] z22;
	 // wire [15:0] z32;
	 // wire [15:0] z42;
	 // wire [15:0] z1;
	 // wire [15:0] z2;
	 // wire [15:0] z3;
	 // wire [15:0] z4;
	 
	 // assign y1 = y?0:16'b0100000000000000;
	 // assign y2 = y?16'b0100000000000000:0;

	 assign pred = (m1 > m2)?1'b1:1'b0;
	 
	 
	 weight_add t1(
	 	.clk(clk),
		.rst_n(rst_n),
		.en(en),
		.w11(w11),
		.w12(w12),
		.w13(w13),
		.w14(w14),
		.w21(w21),
		.w22(w22),
		.w23(w23),
		.w24(w24),
		.h1(h1),
		.h2(h2),
		.h3(h3),
		.h4(h4),
		.result_valid(result_valid),
		.out1(m1),
		.out2(m2)
	);
						
		// softmax_2 t2(.clk(clk),
		// 				 .rst_n(rst_n),
		// 				 .x1(m1),
		// 				 .x2(m2),
		// 				 .output1(p1),
		// 				 .output2(p2)
		// 				 );
						 
		// assign cost_x1 = y1 - p1;
		// assign cost_x2 = y2 - p2;
		
		// multiplier CW11(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x1), 
		// 					.multiplicator2(h1), 
		// 					.result(r11)
		// 					);
							
		// multiplier CW12(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x1), 
		// 					.multiplicator2(h2), 
		// 					.result(r12)
		// 					);
							
		// multiplier CW13(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x1), 
		// 					.multiplicator2(h3), 
		// 					.result(r13)
		// 					);
							
		// multiplier CW14(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x1), 
		// 					.multiplicator2(h4), 
		// 					.result(r14)
		// 					);
							
		// multiplier CW21(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x2), 
		// 					.multiplicator2(h1), 
		// 					.result(r21)
		// 					);
							
		// multiplier CW22(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x2), 
		// 					.multiplicator2(h2), 
		// 					.result(r22)
		// 					);
							
		// multiplier CW23(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x2), 
		// 					.multiplicator2(h3), 
		// 					.result(r23)
		// 					);
							
		// multiplier CW24(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x2), 
		// 					.multiplicator2(h4), 
		// 					.result(r24)
		// 					);
							
		// multiplier CH11(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x1), 
		// 					.multiplicator2(w11), 
		// 					.result(z11)
		// 					);
							
		// multiplier CH12(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x2), 
		// 					.multiplicator2(w21), 
		// 					.result(z12)
		// 					);
							
		// multiplier CH21(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x1), 
		// 					.multiplicator2(w12), 
		// 					.result(z21)
		// 					);
							
		// multiplier CH22(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x2), 
		// 					.multiplicator2(w22), 
		// 					.result(z22)
		// 					);
							
		// multiplier CH31(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x1), 
		// 					.multiplicator2(w13), 
		// 					.result(z31)
		// 					);
							
		// multiplier CH32(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x2), 
		// 					.multiplicator2(w23), 
		// 					.result(z32)
		// 					);
							
		// multiplier CH41(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x1), 
		// 					.multiplicator2(w14), 
		// 					.result(z41)
		// 					);
							
		// multiplier CH42(
		// 					.clk(clk), 
		// 					.rst_n(rst_n),
		// 					.multiplicator1(cost_x2), 
		// 					.multiplicator2(w24), 
		// 					.result(z42)
		// 					);
							
		// assign z1 = z11 + z12;
		// assign z2 = z21 + z22;
		// assign z3 = z31 + z32;
		// assign z4 = z41 + z42;
							
						 
		// always @(*)
		// begin
		// 	cost_w11 = r11;
		// 	cost_w12 = r12;
		// 	cost_w13 = r13;
		// 	cost_w14 = r14;
		// 	cost_w21 = r21;
		// 	cost_w22 = r22;
		// 	cost_w23 = r23;
		// 	cost_w24 = r24;
		// 	cost_h1 = z1;
		// 	cost_h2 = z2;
		// 	cost_h3 = z3;
		// 	cost_h4 = z4;
		// end

		
		
endmodule
