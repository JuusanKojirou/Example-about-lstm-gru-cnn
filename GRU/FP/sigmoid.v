module sigmoid(
	input clk,
	input rst_n,
	input en,
	input [15:0] sigmoid_input,
	output reg result_valid,
	output reg [15:0] sigmoid_output
    );

//take 4 clk cycles 
	wire signal;
	wire [15:0] sigmoid_input_minus;
	wire [15:0] sigmoid_input_p;
	assign sigmoid_input_minus = ~sigmoid_input + 16'd1;
	assign signal = sigmoid_input[15];

	assign sigmoid_input_p = signal?sigmoid_input_minus:sigmoid_input;
	

	wire [15:0] derivative;
	wire [15:0] delta_x;
	wire [15:0] temp;
	wire [15:0] finalresult;


	assign derivative = (sigmoid_input_p>16'b0010000000000000)?((sigmoid_input_p>16'b0100000000000000)?(16'b 0000100110010101):(16'b 0000110111100110)):(16'b0001000000000000);
	assign delta_x = (sigmoid_input_p>16'b0010000000000000)?((sigmoid_input_p>16'b0100000000000000)?(sigmoid_input_p-16'b0100000000000000):(sigmoid_input_p-16'b0010000000000000)):sigmoid_input_p;

	wire rv;
	mult16x16_2int S(
		.clk(clk),
		.rst_n(rst_n),
		.en(en),
		.multiplicator1(derivative),
		.multiplicator2(delta_x),
		.result_valid(rv),
		.result(temp)
		);

	assign finalresult = (sigmoid_input_p>16'b0010000000000000)  ?  ((sigmoid_input_p>16'b0100000000000000) ? (signal?(16'b0100000000000000-temp-16'b0010111011001001):(temp+16'b0010111011001001)) : (signal?(16'b0100000000000000-temp-16'b0010011111010110):(temp+16'b0010011111010110)))  :  (signal?(16'b0010000000000000-temp):(temp+16'b0010000000000000));

	always @(posedge clk or negedge rst_n)begin
		if(~rst_n) begin 
			sigmoid_output <= 0;
			result_valid <= 0;
		end 
		else begin
			sigmoid_output <= finalresult;
			result_valid <= rv;
		end
	end

endmodule
