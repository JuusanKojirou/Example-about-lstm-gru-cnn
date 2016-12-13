module vector_mult_4_parallel(
	input clk,
	input rst_n,
	input en,
	input signed [15:0] a_0,
	input signed [15:0] a_1,
	input signed [15:0] a_2,
	input signed [15:0] a_3,
	input signed [15:0] b_0,
	input signed [15:0] b_1,
	input signed [15:0] b_2,
	input signed [15:0] b_3,
	output wire result_valid,
	output wire signed [15:0] ab_out
);

//take 3 clk cycles

wire signed [15:0] ab_0;
wire signed [15:0] ab_1;
wire signed [15:0] ab_2;
wire signed [15:0] ab_3;

wire result_valid0;
wire result_valid1;
wire result_valid2;
wire result_valid3;


mult16x16_2int M0(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(a_0),
	.multiplicator2(b_0),
	.result_valid(result_valid0),
	.result(ab_0)
	);
mult16x16_2int M1(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(a_1),
	.multiplicator2(b_1),
	.result_valid(result_valid1),
	.result(ab_1)
	);
mult16x16_2int M2(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(a_2),
	.multiplicator2(b_2),
	.result_valid(result_valid2),
	.result(ab_2)
	);
mult16x16_2int M3(
	.clk(clk),
	.rst_n(rst_n),
	.en(en),
	.multiplicator1(a_3),
	.multiplicator2(b_3),
	.result_valid(result_valid3),
	.result(ab_3)
	);


// always @(posedge clk or negedge rst_n) begin
// 	if(~rst_n) begin
// 		ab_out <= 0;
// 	end else begin
// 		ab_out <= ab_0+ab_1+ab_2+ab_3;
// 	end
// end

assign ab_out = ab_0+ab_1+ab_2+ab_3;
assign result_valid = (result_valid0&result_valid1&result_valid2&result_valid3); 

endmodule
