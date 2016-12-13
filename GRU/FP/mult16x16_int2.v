module mult16x16_2int(
	input clk,
	input rst_n,
	input en,
	input [15:0] multiplicator1,
	input [15:0] multiplicator2,
	output wire result_valid,
	output wire [15:0] result
);

//take 3 clk cycles

reg [20:0] longresult;
wire [15:0] trueform1;
assign trueform1 = multiplicator1[15]?((~multiplicator1)+16'd1):(multiplicator1);
wire [20:0] temp1;
assign temp1 = {trueform1,5'b00000};

reg sign1;
reg sign1d1;
reg sign1d2;

reg result_valid0;
reg result_valid1;
reg result_valid2;

reg [20:0] temp1r14;
reg [20:0] temp1r13;
reg [20:0] temp1r12;
reg [20:0] temp1r11;
reg [20:0] temp1r10;
reg [20:0] temp1r9;
reg [20:0] temp1r8;
reg [20:0] temp1r7;
reg [20:0] temp1r6;
reg [20:0] temp1r5;
reg [20:0] temp1r4;
reg [20:0] temp1r3;
reg [20:0] temp1r2;
reg [20:0] temp1r1;
reg [20:0] temp1r0;
reg [20:0] temp1l1;
reg [20:0] result0;
reg [20:0] result1;
reg [20:0] result2;
reg [20:0] result3;

always @(posedge clk or negedge rst_n or negedge en)begin
	if ((~rst_n) | (~en) ) begin
		temp1r14 <= 0;
		temp1r13 <= 0;
		temp1r12 <= 0;
		temp1r11 <= 0;
		temp1r10 <= 0;
		temp1r9 <= 0;
		temp1r8 <= 0;
		temp1r7 <= 0;
		temp1r6 <= 0;
		temp1r5 <= 0;
		temp1r4 <= 0;
		temp1r3 <= 0;
		temp1r2 <= 0;
		temp1r1 <= 0;
		temp1r0 <= 0;
		temp1l1 <= 0;
		sign1 <= 0;
		result_valid0 <= 0;
	end 
	else begin
		temp1r14 <= multiplicator2[0]? (temp1>>>14):(21'd0);
		temp1r13 <= multiplicator2[1]? (temp1>>>13):(21'd0);
		temp1r12 <= multiplicator2[2]? (temp1>>>12):(21'd0);
		temp1r11 <= multiplicator2[3]? (temp1>>>11):(21'd0);
		temp1r10 <= multiplicator2[4]? (temp1>>>10):(21'd0);
		temp1r9 <= multiplicator2[5]? (temp1>>>9):(21'd0);
		temp1r8 <= multiplicator2[6]? (temp1>>>8):(21'd0);
		temp1r7 <= multiplicator2[7]? (temp1>>>7):(21'd0);
		temp1r6 <= multiplicator2[8]? (temp1>>>6):(21'd0);
		temp1r5 <= multiplicator2[9]? (temp1>>>5):(21'd0);
		temp1r4 <= multiplicator2[10]? (temp1>>>4):(21'd0);
		temp1r3 <= multiplicator2[11]? (temp1>>>3):(21'd0);
		temp1r2 <= multiplicator2[12]? (temp1>>>2):(21'd0);
		temp1r1 <= multiplicator2[13]? (temp1>>>1):(21'd0);
		temp1r0 <= multiplicator2[14]? (temp1):(21'd0);
		temp1l1 <= multiplicator2[15]? (temp1<<1):(21'd0);
		sign1 <= multiplicator1[15];
		result_valid0 <= 1;
	end
end

always @(posedge clk or negedge rst_n) begin
	if(~rst_n) begin
		sign1d1 <= 0;
		sign1d2 <= 0;
		result_valid1 <= 0;
		result_valid2 <= 0;
		result0 <= 0;
		result1 <= 0;
		result2 <= 0;
		result3 <= 0;
		longresult <= 0;
	end else begin
		sign1d1 <= sign1;
		sign1d2 <= sign1d1;
		result_valid1 <= result_valid0;
		result_valid2 <= result_valid1;
		result0 <= temp1r14 + temp1r13 + temp1r12 + temp1r11;
		result1 <= temp1r10 + temp1r9 + temp1r8 + temp1r7;
		result2 <= temp1r6 + temp1r5 + temp1r4 + temp1r3;
		result3 <= temp1r2 + temp1r1 + temp1r0 - temp1l1;//2
		longresult <= result0 + result1 + result2 + result3;//3
	end
end


assign result = sign1d2?(~longresult[20:5]+16'd1):(longresult[20:5]);
assign result_valid = result_valid2;
endmodule