`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    09:26:27 12/20/2015 
// Design Name: 
// Module Name:    average 
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
module average(
    input clk,
    input clkl,
    input rst_n,
    input signed [15:0] ht,
    input start,    //input finish,
	input [7:0] t,
	output done_sig,
    output reg signed [15:0] ave_out    
    );
	 
reg signed [21:0] sum;
reg signed [21:0] sum_in;
reg [7:0] count;
reg sum_done;
//wire T;
wire signed [15:0] rem1;
wire signed [15:0] c;

//assign T = {9'b0,t};


always @(posedge clkl or negedge rst_n)
begin
	if(!rst_n)
		begin
			sum <= 0;
			sum_in <= 0;
			sum_done <= 0;
			count <= 0;
		end
	else begin
		// if(finish) begin
		// 	sum <= sum;
		// 	sum_done <= 1;
		// end
		if(start) begin
			if(count==t) begin
				sum_in <= sum;
				sum <= 0;
				sum_done <= 1;
				count <= 0;
			end
			else begin
				sum <= sum + ht;
				sum_in <= 0;
				sum_done <= 0;
				count <= count+8'd1;
			end
		end
		else begin
			sum <= 0;
			sum_in <=0;
			sum_done <= 0;
			count <= 0;
		end
	end
end

divider_22 av(.rst_n(rst_n),
           .clock(clk),
           .start_sig(sum_done),
           .done_sig(done_sig),
           .a(sum_in),
           .b(t),
           .c(c),
           .d(rem1)
           );

always@(*)
begin
		ave_out = c;
end		


endmodule
